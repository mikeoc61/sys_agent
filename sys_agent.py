#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.40",
#     "anthropic>=0.40,<2.0",
#     "gnureadline>=8.1; sys_platform == 'darwin'",
# ]
# ///
"""
sys_agent.py — Minimal multi-provider shell agent (OpenAI / Anthropic).

Gathers host facts at startup, ships them to the selected model so command
syntax matches the actual environment, then executes model-proposed commands
locally only after explicit user approval. Uses native function/tool calling
on both providers (no fragile prose parsing).

Requires: uv  (https://astral.sh/uv) — installs openai + anthropic on first run
Run:      ./sys_agent.py            (after chmod +x)
          uv run sys_agent.py       (explicit form)

Env:      OPENAI_API_KEY            (one of these is required)
          ANTHROPIC_API_KEY
          SYS_PROVIDER               (skip prompt: openai|anthropic)
          SYS_OPENAI_MODEL           (default gpt-4o-mini)
          SYS_ANTHROPIC_MODEL        (default claude-haiku-4-5-20251001)
          SYS_ENV_FILE               (path to env file; overrides search)
          SYS_COLOR                  (on|off|auto, default auto)
          NO_COLOR                  (if set, disables color regardless)

If SYS_ENV_FILE is not set, the script searches these locations in order
and uses the first that exists:
    ./.env
    $XDG_CONFIG_HOME/sys_agent/.env  (default ~/.config/sys_agent/.env)
    ~/.sys_agent.env
Shell-exported vars always override file values.
"""

from __future__ import annotations

# readline must be imported before any SSL-using library (openai, anthropic)
# on macOS to avoid a segfault-on-exit quirk. Prefer gnureadline (proper GNU
# readline as a drop-in for libedit, fixing colored prompts and history
# redraw on macOS). Falls back to stdlib readline (GNU on Linux, libedit on
# macOS without gnureadline). Falls back further to no readline (Windows).
try:
    import gnureadline as readline     # noqa: F401 — preferred backend
    _HAVE_READLINE = True
    _IS_LIBEDIT = False
except ImportError:
    try:
        import readline                # noqa: F401 — stdlib fallback
        _HAVE_READLINE = True
        # macOS stdlib readline links to libedit, which strips ANSI escapes
        # between \001/\002 markers — killing colored prompts and causing
        # prefix redraw issues on history navigation. Detect to degrade
        # gracefully if gnureadline failed to install for any reason.
        _IS_LIBEDIT = bool(readline.__doc__ and "libedit" in readline.__doc__)
    except ImportError:
        _HAVE_READLINE = False
        _IS_LIBEDIT = False

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import atexit
from dataclasses import dataclass
from typing import Any


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

# OpenAI options (verified May 2026):
#   gpt-4.1-nano  — $0.10/$0.40 per 1M tok, weakest tool use of the three
#   gpt-4o-mini   — $0.15/$0.60, generous free-tier RPM, solid tool use
#   gpt-5.4-mini  — $0.75/$4.50, newer reasoning, best tool use of cheap tier
DEFAULT_OPENAI_MODEL = os.environ.get("SYS_OPENAI_MODEL", "gpt-4o-mini")

# Anthropic options (verified May 2026):
#   claude-haiku-4-5-20251001  — fast/cheap, solid tool use
#   claude-sonnet-4-6          — better reasoning, mid-tier price
#   claude-opus-4-7            — flagship, premium
DEFAULT_ANTHROPIC_MODEL = os.environ.get(
    "SYS_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
)

# Optional shell-style env file. Loaded if present; existing env vars win.
# Search order for env file when SYS_ENV_FILE is not set. First hit wins.
# Values are expanded with os.path.expanduser/expandvars at lookup time.
def _default_env_candidates() -> list[str]:
    xdg = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return [
        ".env",                              # CWD (dotenv convention)
        os.path.join(xdg, "sys_agent", ".env"),
        "~/.sys_agent.env",                  # home dotfile fallback
    ]

# How much subprocess output to forward back to the model (chars).
OUTPUT_MAX_CHARS = 8000

# Per-command wall-clock timeout (seconds).
COMMAND_TIMEOUT = 60

# Anthropic requires max_tokens; pick something generous for tool dialogs.
ANTHROPIC_MAX_TOKENS = 4096

# SDK-level retry count. Both the openai and anthropic SDKs implement
# exponential backoff with jitter and honor Retry-After headers; they retry
# on 408/409/429 and 5xx (including Anthropic's 529 overloaded_error).
# Default SDK value is 2; bump for resilience on flaky networks / busy API.
API_MAX_RETRIES = 5

# Meta-command reference — single source of truth for /help and the startup
# banner. (command, description). Keep in sync with the handlers in run_repl.
META_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help",           "Show this command list and current toggle states"),
    ("/info",           "Provider, model, session token usage, host facts"),
    ("/reset",          "Clear conversation history and token counters"),
    ("/auto on|off",    "Skip the per-command approval prompt (deny list still applies)"),
    ("/tokens on|off",  "Toggle the per-turn token-usage line"),
    ("/color on|off",   "Toggle ANSI color output"),
    ("/exit, /quit",    "End the session"),
)

# Known context windows (May 2026). Used for context-% display; unknown models
# fall back to printing absolute token counts. Update as new models ship.
CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI
    "gpt-4.1-nano":  1_000_000,
    "gpt-4o-mini":     128_000,
    "gpt-5.4-mini":    400_000,
    # Anthropic
    "claude-haiku-4-5-20251001": 200_000,
    "claude-sonnet-4-6":         200_000,
    "claude-opus-4-7":           200_000,
}

# Local hard-deny list — never executed regardless of provider or user OK.
DENY_SUBSTRINGS = (
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    ":(){ :|:& };:",            # fork bomb
    "dd if=/dev/zero of=/dev/",
    "dd if=/dev/random of=/dev/",
    "> /dev/sda",
    "> /dev/nvme",
    "chmod -R 777 /",
)

# readline history settings
HISTORY_FILE_DEFAULT = "~/.config/sys_agent/history"
HISTORY_MAX_LINES = 1000


# -----------------------------------------------------------------------------
# Color (ANSI escapes, no external deps)
# -----------------------------------------------------------------------------

class _ANSI:
    RESET     = "\033[0m"
    BOLD      = "\033[1m"
    DIM       = "\033[2m"
    RED       = "\033[31m"
    GREEN     = "\033[32m"
    YELLOW    = "\033[33m"
    CYAN      = "\033[36m"
    BR_YELLOW = "\033[93m"

# Module-level toggle, set by init_color() and /color REPL command.
_color_enabled: bool = False


def init_color() -> None:
    """Decide whether to emit ANSI. Honors NO_COLOR, SYS_COLOR, TTY status."""
    global _color_enabled
    pref = os.environ.get("SYS_COLOR", "auto").strip().lower()
    if pref == "off":
        _color_enabled = False
        return
    if pref == "on":
        _color_enabled = True
        return
    # auto
    if "NO_COLOR" in os.environ:
        _color_enabled = False
        return
    if not sys.stdout.isatty():
        _color_enabled = False
        return
    if os.environ.get("TERM", "") == "dumb":
        _color_enabled = False
        return
    _color_enabled = True


def _wrap(text: str, *codes: str) -> str:
    if not _color_enabled or not codes:
        return text
    return "".join(codes) + text + _ANSI.RESET


# Semantic helpers — call sites read clearly, palette stays centralized.
def dim(t: str) -> str:        return _wrap(t, _ANSI.DIM)
def ok(t: str) -> str:         return _wrap(t, _ANSI.GREEN)
def fail(t: str) -> str:       return _wrap(t, _ANSI.RED)
def warn(t: str) -> str:       return _wrap(t, _ANSI.YELLOW)
def warn_bold(t: str) -> str:  return _wrap(t, _ANSI.BOLD, _ANSI.YELLOW)
def err_bold(t: str) -> str:   return _wrap(t, _ANSI.BOLD, _ANSI.RED)
def cmd_label(t: str) -> str:  return _wrap(t, _ANSI.BOLD, _ANSI.BR_YELLOW)
def banner(t: str) -> str:     return _wrap(t, _ANSI.BOLD, _ANSI.CYAN)
def ask(t: str) -> str:        return _wrap(t, _ANSI.BOLD, _ANSI.CYAN)
def user_tag(t: str) -> str:   return _wrap(t, _ANSI.BOLD, _ANSI.CYAN)
def agent_tag(t: str) -> str:  return _wrap(t, _ANSI.BOLD, _ANSI.GREEN)


def rl_prompt(text: str) -> str:
    """
    Wrap a colored prompt for safe use with input() on GNU readline.
    Markers \\001 and \\002 tell readline "this region is zero-width" — fixing
    both cursor math and history-redraw behavior. On libedit (macOS default)
    the markers cause ANSI to be stripped entirely, so callers should use
    colored_input() instead of this helper directly.

    No-op when color is disabled or readline is unavailable.
    """
    if not _color_enabled or not _HAVE_READLINE:
        return text
    return re.sub(r"(\x1b\[[0-9;]*m)", "\001\\1\002", text)


def colored_input(prompt: str) -> str:
    """
    Read a line of input with a colored prompt, choosing the right strategy
    based on the readline backend:

    - GNU readline (Linux/Pi): pass colored text through input() with
      \\001/\\002 width-hint markers. Color renders; cursor math correct;
      history Up-arrow redraws cleanly.
    - libedit (macOS default): markers strip ANSI, so print the colored
      prefix separately and call input(""). Color renders; cursor math
      correct (empty prompt = 0 width). Tradeoff: history Up-arrow may
      visually overwrite the prefix in some terminals — cosmetic only.
    - No readline (Windows): plain input(prompt).
    """
    if _color_enabled and _IS_LIBEDIT:
        print(prompt, end="", flush=True)
        return input("")
    return input(rl_prompt(prompt))


def drop_last_history_entry() -> None:
    """Remove the most recent readline history entry. Safe no-op without readline."""
    if not _HAVE_READLINE:
        return
    try:
        length = readline.get_current_history_length()
        if length > 0:
            readline.remove_history_item(length - 1)
    except (ValueError, IndexError):
        # Empty entry never added, or backend quirk — safe to ignore.
        pass


def input_no_history(prompt: str) -> str:
    """
    colored_input variant that drops the entered line from history. Used for
    short-answer prompts (y/n/edit/quit, provider choice) that would only
    pollute Up-arrow recall of real conversational prompts.
    """
    ans = colored_input(prompt)
    drop_last_history_entry()
    return ans


# -----------------------------------------------------------------------------
# readline (input history + line editing)
# -----------------------------------------------------------------------------

# Module-level path so /reset etc. can reference if needed later.
_history_path: str | None = None


def _decode_libedit_escapes() -> None:
    """
    libedit writes history with octal escapes for whitespace/backslash
    (\\040 for space, \\011 for tab, \\134 for backslash). gnureadline reads
    those as literal text, so a file written by libedit and loaded by
    gnureadline shows "How\\040does\\040swap?" on Up-arrow recall.

    Walk the loaded history once on startup; decode any entry that contains
    libedit-style escapes. Idempotent — entries already plain pass through
    unchanged. Called by init_readline after read_history_file.
    """
    if not _HAVE_READLINE:
        return
    n = readline.get_current_history_length()
    for i in range(1, n + 1):
        item = readline.get_history_item(i)
        if not item or "\\0" not in item and "\\1" not in item:
            continue
        decoded = (item
                   .replace("\\040", " ")
                   .replace("\\011", "\t")
                   .replace("\\012", "\n")
                   .replace("\\134", "\\"))
        if decoded != item:
            readline.replace_history_item(i - 1, decoded)


def init_readline() -> None:
    """
    Configure readline history persistence and length cap. Safe no-op when
    readline is unavailable (Windows). Failures are non-fatal — printed once,
    then the REPL continues without history persistence.
    """
    global _history_path
    if not _HAVE_READLINE:
        return
    path = os.path.expanduser(HISTORY_FILE_DEFAULT)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError as e:
        print(dim(f"[readline: cannot create history dir: {e}]"))
        return
    # Load prior history if present; missing file is fine.
    try:
        readline.read_history_file(path)
        _decode_libedit_escapes()
    except FileNotFoundError:
        pass
    except OSError as e:
        print(dim(f"[readline: cannot read history: {e}]"))
    # Cap history length on disk (in-memory cap set separately).
    try:
        readline.set_history_length(HISTORY_MAX_LINES)
    except Exception:           # noqa: BLE001 — some libedit builds are picky
        pass
    _history_path = path


def save_readline_history() -> None:
    """Persist history on exit. Non-fatal on failure."""
    if not _HAVE_READLINE or _history_path is None:
        return
    try:
        readline.write_history_file(_history_path)
    except OSError as e:
        print(dim(f"[readline: cannot save history: {e}]"))


# -----------------------------------------------------------------------------
# Env file loader (shell-style KEY=value)
# -----------------------------------------------------------------------------

def load_env_file(path: str) -> int:
    """
    Load shell-style env file into os.environ. Existing env vars win.
    Silent no-op if file missing. Returns count of vars set.
    """
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        return 0
    loaded = 0
    with open(expanded, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val
                loaded += 1
    return loaded


def find_env_file(explicit: str | None) -> str | None:
    """
    Return the first existing env-file path. If `explicit` is set, only that
    path is tried (returning None if missing, so the caller can warn).
    Otherwise the default candidate list is searched in priority order.
    """
    candidates = [explicit] if explicit else _default_env_candidates()
    for p in candidates:
        if not p:
            continue
        expanded = os.path.expanduser(os.path.expandvars(p))
        if os.path.isfile(expanded):
            return expanded
    return None


# -----------------------------------------------------------------------------
# Host fact-gathering
# -----------------------------------------------------------------------------

def _safe_read(path: str, limit: int = 2048) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(limit).strip()
    except OSError:
        return ""


def _which_many(cmds: list[str]) -> list[str]:
    return [c for c in cmds if shutil.which(c)]


def gather_host_facts() -> dict[str, Any]:
    uname = platform.uname()
    facts: dict[str, Any] = {
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "node": uname.node,
        "python": platform.python_version(),
        "shell": os.environ.get("SHELL", "unknown"),
        "user": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
        "cwd": os.getcwd(),
        "home": os.path.expanduser("~"),
        "lang": os.environ.get("LANG", ""),
        "term": os.environ.get("TERM", ""),
    }
    if uname.system == "Linux":
        kv: dict[str, str] = {}
        for line in _safe_read("/etc/os-release").splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                kv[k.strip()] = v.strip().strip('"')
        facts["distro_id"] = kv.get("ID", "")
        facts["distro_pretty"] = kv.get("PRETTY_NAME", "")
        facts["distro_version_id"] = kv.get("VERSION_ID", "")
    if uname.system == "Darwin":
        facts["mac_version"] = platform.mac_ver()[0]
    candidates = [
        # package managers
        "apt", "apt-get", "dnf", "yum", "pacman", "zypper", "brew", "pkg",
        # init / services
        "systemctl", "service", "launchctl", "rc-service",
        # network
        "ip", "ifconfig", "ss", "netstat", "iptables", "nft", "nmcli", "dig", "host",
        # editors
        "vim", "nvim", "nano", "emacs",
        # core utilities
        "curl", "wget", "git", "jq", "rsync", "tmux", "screen", "tree",
        # containers / VMs
        "docker", "podman", "kubectl",
        # python ecosystem
        "python3", "pip", "pip3", "uv", "poetry", "pipx",
    ]
    facts["available_tools"] = _which_many(candidates)
    return facts


# -----------------------------------------------------------------------------
# Tool schema (shared core, provider-specific wrappers)
# -----------------------------------------------------------------------------

TOOL_NAME = "run_command"
TOOL_DESCRIPTION = (
    "Execute a shell command on the user's local machine and return stdout, "
    "stderr, and exit code. Use this both to inspect state and to make changes. "
    "Every call is shown to the user and requires approval before execution; "
    "design each command to be self-contained and, where reasonable, idempotent."
)
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": (
                "Exact command as it would be typed into the shell. "
                "Pipes, redirects, and shell builtins are allowed."
            ),
        },
        "explanation": {
            "type": "string",
            "description": (
                "One short sentence: what this does and why. "
                "Shown to the user during approval."
            ),
        },
    },
    "required": ["command", "explanation"],
    "additionalProperties": False,
}

OPENAI_TOOLS = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": TOOL_PARAMETERS,
    },
}]

ANTHROPIC_TOOLS = [{
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "input_schema": TOOL_PARAMETERS,
}]


# -----------------------------------------------------------------------------
# Command execution
# -----------------------------------------------------------------------------

@dataclass
class CmdResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    truncated: bool = False

    def as_tool_payload(self) -> str:
        out, err = self.stdout, self.stderr
        if len(out) > OUTPUT_MAX_CHARS:
            out = out[:OUTPUT_MAX_CHARS] + "\n...[stdout truncated]"
            self.truncated = True
        if len(err) > OUTPUT_MAX_CHARS:
            err = err[:OUTPUT_MAX_CHARS] + "\n...[stderr truncated]"
            self.truncated = True
        return json.dumps({
            "returncode": self.returncode,
            "stdout": out,
            "stderr": err,
            "truncated": self.truncated,
        })


def is_denied(cmd: str) -> str | None:
    for needle in DENY_SUBSTRINGS:
        if needle in cmd:
            return f"matches local deny rule: {needle!r}"
    return None


def prompt_approval(cmd: str, explanation: str, auto: bool) -> tuple[str, str | None]:
    """Returns (action, edited_cmd_or_none). action in {run, skip, abort}."""
    print()
    print(dim("─" * 72))
    # Command body stays plain (per preference); only the label is loud.
    print(f"{cmd_label('COMMAND:')}  {cmd}")
    print(f"REASON:   {explanation}")
    print(dim(f"CWD:      {os.getcwd()}"))
    print(dim("─" * 72))
    if auto:
        print(warn("[auto-approve on] running."))
        return "run", None
    while True:
        try:
            ans = input_no_history(warn_bold("Run? [y]es / [n]o / [e]dit / [q]uit: ")).strip().lower()
        except EOFError:
            return "abort", None
        if ans in ("y", "yes", ""):
            return "run", None
        if ans in ("n", "no"):
            return "skip", None
        if ans in ("q", "quit", "abort"):
            return "abort", None
        if ans in ("e", "edit"):
            # edit prompt plain — it's a user input position
            try:
                new = input("edit> ").strip()
            except EOFError:
                return "abort", None
            if new:
                return "run", new


def execute(cmd: str) -> CmdResult:
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=COMMAND_TIMEOUT
        )
        return CmdResult(cmd, proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as e:
        return CmdResult(
            cmd, 124,
            (e.stdout or "") if isinstance(e.stdout, str) else "",
            f"TIMEOUT after {COMMAND_TIMEOUT}s",
        )
    except Exception as e:                  # noqa: BLE001
        return CmdResult(cmd, 1, "", f"exec error: {e}")


# -----------------------------------------------------------------------------
# Provider abstraction
# -----------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Usage:
    input_tokens: int = 0       # tokens sent THIS call (≈ current context size)
    output_tokens: int = 0      # tokens generated THIS call


@dataclass
class ChatTurn:
    text: str
    tool_calls: list[ToolCall]
    raw_message: Any           # provider-native dict, appended to history
    usage: Usage


class Provider:
    name: str = ""
    model: str = ""

    def initial_messages(self, system: str) -> list[dict]:
        raise NotImplementedError

    def chat(self, messages: list[dict], system: str) -> ChatTurn:
        raise NotImplementedError

    def append_assistant(self, messages: list[dict], turn: ChatTurn) -> None:
        messages.append(turn.raw_message)

    def append_tool_results(
        self, messages: list[dict], results: list[tuple[str, str]]
    ) -> None:
        raise NotImplementedError


class OpenAIProvider(Provider):
    def __init__(self, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("openai SDK missing — uv should install via PEP 723 deps")
        self.client = OpenAI(max_retries=API_MAX_RETRIES)
        self.model = model
        self.name = "openai"

    def initial_messages(self, system: str) -> list[dict]:
        # OpenAI: system prompt is the first message
        return [{"role": "system", "content": system}]

    def chat(self, messages: list[dict], system: str) -> ChatTurn:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return ChatTurn(
            text=msg.content or "",
            tool_calls=calls,
            raw_message=msg.model_dump(exclude_none=True),
            usage=Usage(
                input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            ),
        )

    def append_tool_results(
        self, messages: list[dict], results: list[tuple[str, str]]
    ) -> None:
        # OpenAI: one `role: tool` message per tool_call_id
        for tc_id, content in results:
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": content,
            })


class AnthropicProvider(Provider):
    def __init__(self, model: str) -> None:
        try:
            from anthropic import Anthropic
        except ImportError:
            sys.exit("anthropic SDK missing — uv should install via PEP 723 deps")
        self.client = Anthropic(max_retries=API_MAX_RETRIES)
        self.model = model
        self.name = "anthropic"

    def initial_messages(self, system: str) -> list[dict]:
        # Anthropic: system prompt is a top-level parameter, not in messages
        return []

    def chat(self, messages: list[dict], system: str) -> ChatTurn:
        # Send the system prompt as a cacheable block. The system prompt
        # (instructions + host-facts JSON) is identical every turn, so
        # marking it with cache_control lets Anthropic bill cache reads at
        # ~10% of input rate after the first call. Cache TTL is 5 min,
        # refreshed on each hit — fine for an interactive session.
        system_blocks = [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
        resp = self.client.messages.create(
            model=self.model,
            system=system_blocks,
            messages=messages,
            tools=ANTHROPIC_TOOLS,
            max_tokens=ANTHROPIC_MAX_TOKENS,
        )
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        raw_content: list[dict] = []
        for block in resp.content:
            # exclude_none keeps the echo-back payload minimal and valid
            raw_content.append(block.model_dump(exclude_none=True))
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.input or {}),
                ))
        return ChatTurn(
            text="\n".join(text_parts),
            tool_calls=calls,
            raw_message={"role": "assistant", "content": raw_content},
            usage=Usage(
                input_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
                output_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
            ),
        )

    def append_tool_results(
        self, messages: list[dict], results: list[tuple[str, str]]
    ) -> None:
        # Anthropic: ALL tool_result blocks for one assistant turn go in
        # a single user message; mismatched batches are rejected.
        blocks = [
            {"type": "tool_result", "tool_use_id": tc_id, "content": content}
            for tc_id, content in results
        ]
        messages.append({"role": "user", "content": blocks})


# -----------------------------------------------------------------------------
# Provider selection
# -----------------------------------------------------------------------------

def select_provider() -> Provider:
    have_openai = bool(os.environ.get("OPENAI_API_KEY"))
    have_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not have_openai and not have_anthropic:
        sys.exit("error: neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set")

    # Honor SYS_PROVIDER override (skips interactive prompt)
    pref = os.environ.get("SYS_PROVIDER", "").strip().lower()
    if pref in ("openai", "o"):
        if not have_openai:
            sys.exit("error: SYS_PROVIDER=openai but OPENAI_API_KEY not set")
        return OpenAIProvider(DEFAULT_OPENAI_MODEL)
    if pref in ("anthropic", "a", "claude"):
        if not have_anthropic:
            sys.exit("error: SYS_PROVIDER=anthropic but ANTHROPIC_API_KEY not set")
        return AnthropicProvider(DEFAULT_ANTHROPIC_MODEL)

    # Only one key available — no prompt needed
    if have_openai and not have_anthropic:
        print(dim("[only OPENAI_API_KEY available — using OpenAI]"))
        return OpenAIProvider(DEFAULT_OPENAI_MODEL)
    if have_anthropic and not have_openai:
        print(dim("[only ANTHROPIC_API_KEY available — using Anthropic]"))
        return AnthropicProvider(DEFAULT_ANTHROPIC_MODEL)

    # Both available — prompt interactively
    print(banner("Select LLM provider:"))
    print(f"  [1] OpenAI     (model: {DEFAULT_OPENAI_MODEL})")
    print(f"  [2] Anthropic  (model: {DEFAULT_ANTHROPIC_MODEL})")
    while True:
        try:
            ans = input_no_history(ask("Choice [1/2]: ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nno selection; exiting")
        if ans in ("1", "o", "openai"):
            return OpenAIProvider(DEFAULT_OPENAI_MODEL)
        if ans in ("2", "a", "anthropic", "claude"):
            return AnthropicProvider(DEFAULT_ANTHROPIC_MODEL)


# -----------------------------------------------------------------------------
# REPL
# -----------------------------------------------------------------------------

def explain_api_error(e: Exception) -> str:
    """
    Turn a raw SDK exception into a short, actionable message. The SDK has
    already exhausted API_MAX_RETRIES by the time this is called, so these
    are persistent failures, not transient blips.
    """
    s = str(e)
    # status_code attribute is present on both SDKs' APIStatusError subclasses
    code = getattr(e, "status_code", None)
    if code == 529 or "overloaded" in s.lower():
        return ("API overloaded (529) — the provider is at capacity. "
                "Retries were exhausted. Wait a minute and resend, or try "
                "the other provider (/exit and relaunch).")
    if code == 429 or "rate limit" in s.lower():
        return ("Rate limited (429) — you've hit a usage tier limit. "
                "Wait before resending, or check your plan limits.")
    if code == 401 or "authentication" in s.lower() or "api key" in s.lower():
        return ("Authentication failed (401) — the API key is missing, "
                "invalid, or expired. Check your env file / shell env.")
    if code in (500, 502, 503, 504):
        return (f"Provider server error ({code}) — transient on their end. "
                "Retries were exhausted; resend shortly.")
    if "connection" in s.lower() or "timeout" in s.lower():
        return ("Network error reaching the API — check connectivity "
                "(the Pi's wlan0 power-save quirk can cause this).")
    return s          # fall back to the raw message for anything unrecognized


def print_help(auto_approve: bool, show_tokens: bool) -> None:
    """Print the meta-command reference plus current toggle states."""
    print(banner("Meta-commands:"))
    width = max(len(cmd) for cmd, _ in META_COMMANDS)
    for cmd, desc in META_COMMANDS:
        print(f"  {cmd.ljust(width)}   {dim(desc)}")
    print()
    print(dim(
        f"  state: auto-approve={auto_approve}  "
        f"show-tokens={show_tokens}  color={_color_enabled}"
    ))


def build_system_prompt(facts: dict[str, Any]) -> str:
    return textwrap.dedent(f"""
        You are a command-line assistant operating directly on the user's host.

        HOST CONTEXT (authoritative — match command syntax to this environment):
        {json.dumps(facts, indent=2)}

        Rules:
        - Use the run_command tool to gather info OR make changes; do not
          fabricate output.
        - Tailor every command to the host's OS, distro, shell, and the
          installed tools listed in `available_tools`. Do not invoke a
          package manager or utility that is not present on the host.
        - Prefer non-destructive, read-only commands first. Verify
          assumptions about file paths, services, and versions by inspecting
          the system before making changes.
        - One logical step per tool call. Avoid chaining unrelated commands
          on one line; it makes failures hard to diagnose.
        - Each command runs in a SEPARATE fresh shell. A `cd` in one command
          does NOT carry over to the next. To operate in a directory, either
          use absolute paths, or combine the cd into the same command
          (e.g. `cd /var/log && ls -la`).
        - Commands run with captured stdout/stderr (no tty). Prefer
          script-stable tooling: `apt-get` over `apt` on Debian/Ubuntu (apt
          prints a CLI-stability warning when not on a tty); `dnf` is fine
          on Fedora/RHEL. Add `--no-progress`, `-q`, or equivalent flags
          when a tool offers them and the noise isn't useful.
        - After receiving command output, summarize the relevant findings
          and decide the next step.
        - When the task is complete, or you need user input, reply in plain
          text without calling a tool.
        - Never propose commands that wipe disks, format filesystems, or
          irrecoverably destroy data. If such a step is genuinely required,
          flag it in plain text and ask the user to run it manually.
    """).strip()


def run_repl(provider: Provider) -> None:
    global _color_enabled
    facts = gather_host_facts()
    system = build_system_prompt(facts)
    messages = provider.initial_messages(system)
    auto_approve = False
    show_tokens = False                  # /tokens on|off
    session_in = 0                       # cumulative input tokens this session
    session_out = 0                      # cumulative output tokens this session
    last_in = 0                          # input tokens on most recent call
    last_out = 0                         # output tokens on most recent call
    ctx_window = CONTEXT_WINDOWS.get(provider.model)   # may be None

    def fmt_tokens(this_in: int, this_out: int) -> str:
        ctx_part = (
            f"{this_in}/{ctx_window} ({100 * this_in / ctx_window:.1f}%)"
            if ctx_window else f"{this_in} (window unknown)"
        )
        return dim(
            f"[tokens turn: in={this_in} out={this_out}  "
            f"session: in={session_in} out={session_out}  "
            f"ctx: {ctx_part}]"
        )

    print(banner(
        f"sys_agent  provider={provider.name}  model={provider.model}  "
        f"host={facts['node']} ({facts['system']}/{facts['machine']})"
    ))
    print(dim("meta: " + "  ".join(cmd for cmd, _ in META_COMMANDS)
              + "   — /help for details"))
    print()

    while True:
        try:
            user_in = colored_input(user_tag("you>") + " ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_in:
            continue

        # Meta-commands shouldn't pollute Up-arrow recall of real prompts
        if user_in.startswith("/"):
            drop_last_history_entry()

        # REPL meta-commands
        if user_in in ("/exit", "/quit"):
            return
        if user_in in ("/help", "/?"):
            print_help(auto_approve, show_tokens)
            continue
        if user_in == "/reset":
            messages = provider.initial_messages(system)
            session_in = session_out = last_in = last_out = 0
            print(dim("[conversation reset, token counters cleared]"))
            continue
        if user_in == "/info":
            ctx_str = (
                f"{last_in}/{ctx_window} ({100 * last_in / ctx_window:.1f}%)"
                if ctx_window and last_in
                else (str(ctx_window) if ctx_window else "unknown")
            )
            print(json.dumps({
                "provider": provider.name,
                "model": provider.model,
                "context_window": ctx_window,
                "session": {
                    "input_tokens": session_in,
                    "output_tokens": session_out,
                    "last_call_input_tokens": last_in,
                    "last_call_output_tokens": last_out,
                    "context_used_last_call": ctx_str,
                },
                "auto_approve": auto_approve,
                "show_tokens": show_tokens,
                "color": _color_enabled,
                "host": facts,
            }, indent=2))
            continue
        if user_in.startswith("/auto"):
            parts = user_in.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                auto_approve = parts[1] == "on"
                print(dim(f"[auto-approve = {auto_approve}]"))
            else:
                print(dim(f"[auto-approve = {auto_approve}]  usage: /auto on|off"))
            continue
        if user_in.startswith("/tokens"):
            parts = user_in.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                show_tokens = parts[1] == "on"
                print(dim(f"[show_tokens = {show_tokens}]"))
            else:
                # Bare /tokens prints current snapshot regardless of toggle
                print(fmt_tokens(last_in, last_out))
                print(dim(f"  show_tokens = {show_tokens}  (usage: /tokens on|off)"))
            continue
        if user_in.startswith("/color"):
            parts = user_in.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                _color_enabled = parts[1] == "on"
                print(dim(f"[color = {_color_enabled}]"))
            else:
                print(dim(f"[color = {_color_enabled}]  usage: /color on|off"))
            continue

        messages.append({"role": "user", "content": user_in})

        # Inner loop: keep calling the model until it stops requesting tools.
        iteration = 0
        while True:
            try:
                turn = provider.chat(messages, system)
            except Exception as e:          # noqa: BLE001
                print(err_bold(f"[api error] {explain_api_error(e)}"))
                # Only pop on first iteration (orphan user msg);
                # on later iterations the conversation is in a valid state.
                if iteration == 0:
                    messages.pop()
                break
            iteration += 1

            # Accumulate token usage from this API call
            last_in = turn.usage.input_tokens
            last_out = turn.usage.output_tokens
            session_in += last_in
            session_out += last_out
            if show_tokens:
                print(fmt_tokens(last_in, last_out))

            provider.append_assistant(messages, turn)

            if not turn.tool_calls:
                if turn.text:
                    # Prefix colored, body plain (per preference)
                    print(f"\n{agent_tag('agent>')} {turn.text}\n")
                break

            results: list[tuple[str, str]] = []
            aborted = False
            for tc in turn.tool_calls:
                if aborted:
                    # Anthropic requires a tool_result for every tool_use_id
                    # in the prior assistant turn; OpenAI is lenient but
                    # consistent state is cleaner.
                    results.append((tc.id, json.dumps({"error": "session aborted"})))
                    continue

                if tc.name != TOOL_NAME:
                    results.append((tc.id, json.dumps({"error": f"unknown tool: {tc.name}"})))
                    continue

                cmd = (tc.arguments.get("command") or "").strip()
                why = (tc.arguments.get("explanation") or "").strip() or "(no explanation)"

                if not cmd:
                    results.append((tc.id, json.dumps({"error": "empty command"})))
                    continue

                deny = is_denied(cmd)
                if deny:
                    # blocked tag loud; command body and reason in normal red
                    print(f"\n{err_bold('[blocked]')} {cmd}")
                    print(fail(f"   reason: {deny}"))
                    results.append((tc.id, json.dumps({"error": f"blocked locally: {deny}"})))
                    continue

                action, edited = prompt_approval(cmd, why, auto_approve)
                if action == "abort":
                    results.append((tc.id, json.dumps({"error": "user aborted"})))
                    aborted = True
                    continue
                if action == "skip":
                    results.append((tc.id, json.dumps({"error": "user declined to run command"})))
                    continue

                to_run = edited or cmd
                if edited:
                    # tag colored, edited command body plain
                    print(f"{warn('[running edited]:')} {to_run}")

                result = execute(to_run)
                exit_str = f"[exit={result.returncode}]"
                print(ok(exit_str) if result.returncode == 0 else fail(exit_str))
                if result.stdout:
                    # stdout preserved raw — may contain its own ANSI
                    print(result.stdout.rstrip())
                if result.stderr:
                    print(warn("[stderr]"))
                    print(result.stderr.rstrip())

                results.append((tc.id, result.as_tool_payload()))

            provider.append_tool_results(messages, results)

            if aborted:
                print(fail("[aborted]"))
                return
            # otherwise loop back to feed tool results into the next chat call


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    explicit = os.environ.get("SYS_ENV_FILE")
    env_file = find_env_file(explicit)
    init_color()
    init_readline()
    atexit.register(save_readline_history)
    if env_file:
        n = load_env_file(env_file)
        if n:
            print(dim(f"[loaded {n} vars from {env_file}]"))
    elif explicit:
        print(warn(f"[SYS_ENV_FILE={explicit} not found; relying on shell env]"))
    provider = select_provider()
    run_repl(provider)


if __name__ == "__main__":
    main()
