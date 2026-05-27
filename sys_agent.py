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
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import atexit
from dataclasses import dataclass
from typing import Any


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

# =============================================================================
# PROVIDER & MODEL CONFIG
# -----------------------------------------------------------------------------
# Single source of truth for everything that changes on an LLM-release cadence.
# When a new model ships, edit ONLY this section:
#   1. add it to PROVIDER_MODELS under its provider,
#   2. add a CONTEXT_WINDOWS entry so the context-% display works,
#   3. if it introduces a new name prefix, extend PROVIDER_MODEL_PREFIXES.
# Nothing below this section needs to change for a model update.
# =============================================================================

# Default model per provider, overridable via env. The default is intentionally
# the cheapest/fastest tier; switch at runtime with /model.
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

# Known context windows (May 2026). Used for the context-% display; unknown
# models fall back to printing absolute token counts.
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

# Known model names per provider — the suggestion list for /model and the
# basis for sanity-checking. A name in this list switches silently; an
# unlisted name whose prefix still matches the provider (see
# PROVIDER_MODEL_PREFIXES) switches with a warning, so brand-new releases
# work before this list is updated.
PROVIDER_MODELS: dict[str, tuple[str, ...]] = {
    "openai": (
        "gpt-4o-mini",
        "gpt-4.1-nano",
        "gpt-5.4-mini",
    ),
    "anthropic": (
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
    ),
}

# Model-name prefixes that legitimately belong to each provider. Used to
# accept an unlisted-but-plausible model (with a warning) while still
# rejecting a wrong-provider or nonsense name outright.
PROVIDER_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-", "o1", "o3", "o4", "chatgpt-"),
    "anthropic": ("claude-",),
}

# =============================================================================
# RUNTIME TUNING
# =============================================================================

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

# Per-command wall-clock timeout (seconds). Override with SYS_COMMAND_TIMEOUT.
# Default raised to 120s so package upgrades on slower hosts (e.g. a Pi) are
# less likely to be killed mid-operation. On timeout the whole process group
# is signalled (see execute()), so a killed apt-get does not orphan dpkg.
COMMAND_TIMEOUT = int(os.environ.get("SYS_COMMAND_TIMEOUT", "120"))

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
    ("/provider [name]", "Show or switch provider: openai|anthropic"),
    ("/model [name]",    "Switch model for the active provider; no arg lists choices"),
    ("/facts",           "Print current host facts"),
    ("/facts refresh",   "Re-probe host facts and rebuild system prompt"),
    ("/facts verbose on|off", "Toggle expanded host fact collection"),
    ("/exit, /quit",    "End the session"),
)

# Local hard-deny tables — these commands are never executed regardless of
# provider or user approval. This is a backstop, not a security boundary: the
# per-command approval prompt is the real gate. Matching is intent-based
# (tokenised argv inspection), so a non-root path argument is allowed
# (`rm -rf /home/x` runs) while the catastrophic shape is blocked
# (`rm -rf /`, `rm -fr /`, `sudo rm -rf /*` do not).

# Fork bomb is matched against the WHOLE command — its separators are part
# of the construct, so it must not be split into segments first.
_FORKBOMB_RE = re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&?\s*\}\s*;\s*:")

# Shell segment separators: ; newline && || | and a bare & (but not >& / 2>&1).
_DENY_SEG_RE = re.compile(r";|\n|&&|\|\||\||(?<![>\d])&(?!>)")

# Wrapper commands whose real target is a later token.
_DENY_WRAPPERS = frozenset((
    "sudo", "doas", "env", "nice", "ionice", "nohup", "time",
    "command", "exec", "stdbuf", "setsid", "timeout",
))
# Wrapper options that consume the following token as their value.
_DENY_OPTS_WITH_VALUE = frozenset((
    "-u", "--user", "-g", "--group", "-n", "-c", "-C", "--chdir",
    "-k", "--signal", "-s", "--kill-after",
))

_DENY_RECURSIVE_SHORT = re.compile(r"^-[a-zA-Z]*[rR][a-zA-Z]*$")
_DENY_ROOT_TARGETS = frozenset(("/", "/*", "/.", "//"))
_DENY_BLOCKDEV_RE = re.compile(r"/dev/(?:sd|nvme|mmcblk|vd|hd|disk|loop)\w*\Z")
_DENY_OF_DEV_RE = re.compile(
    r"\Aof=/dev/(?:sd|nvme|mmcblk|vd|hd|disk|loop)\w*", re.I)
_DENY_REDIR_RE = re.compile(r"\A\d?>>?\|?(.*)\Z")

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
    colored_input variant for short-answer prompts (y/n/edit/quit, provider
    choice) that should not land in Up-arrow recall.

    When auto-history is disabled (the normal case — see init_readline),
    nothing is added implicitly, so simply not calling add_history is enough
    and there is nothing to drop. The drop is kept only as a fallback for
    backends lacking set_auto_history, where input() still auto-adds.
    """
    ans = colored_input(prompt)
    if _HAVE_READLINE and not hasattr(readline, "set_auto_history"):
        drop_last_history_entry()
    return ans


def _input_prefilled(prompt: str, text: str) -> str:
    """
    input() with the readline line buffer pre-populated with `text`, so the
    user edits the string in place instead of retyping it. Requires GNU
    readline (set_pre_input_hook); libedit and the no-readline path fall back
    to a bare prompt, leaving the command visible on the COMMAND: line above.
    """
    if (_HAVE_READLINE and not _IS_LIBEDIT
            and hasattr(readline, "set_pre_input_hook")):
        def _hook() -> None:
            readline.insert_text(text)
            readline.redisplay()
        readline.set_pre_input_hook(_hook)
        try:
            return input(prompt)
        finally:
            readline.set_pre_input_hook(None)
    return input(prompt)


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
    # Disable input()'s implicit history-add. It is unreliable across
    # backends: stdlib GNU readline on Linux silently skips the add when the
    # prompt carries \001/\002 width markers (as colored prompts do), so
    # typed commands never reached history on the Pi. With auto-history off,
    # history is added explicitly (see run_repl) — deterministic everywhere.
    # libedit / older builds lack set_auto_history; there the implicit add
    # still happens and input_no_history's drop handles short prompts.
    if hasattr(readline, "set_auto_history"):
        try:
            readline.set_auto_history(False)
        except Exception:       # noqa: BLE001
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


# Pseudo / virtual filesystems and device classes we never want to surface.
# Mirrors disk_smart.NO_SMART_PREFIXES plus filesystem-level pseudos.
_DISK_SKIP_DEV_PREFIXES = ("loop", "ram", "zram", "dm-", "sr", "fd")
_DISK_SKIP_FSTYPES = {
    "proc", "sysfs", "devtmpfs", "devpts", "tmpfs", "cgroup", "cgroup2",
    "pstore", "bpf", "tracefs", "debugfs", "configfs", "fusectl",
    "securityfs", "mqueue", "hugetlbfs", "autofs", "rpc_pipefs",
    "binfmt_misc", "nsfs", "overlay", "squashfs", "ramfs",
}
# Mount-point prefixes that are noise for an agent (snap loops, container
# layer mounts, init-time runtime dirs).
_DISK_SKIP_MOUNT_PREFIXES = (
    "/snap/", "/var/lib/docker/", "/var/lib/containers/",
    "/run/", "/sys/", "/proc/", "/dev/",
)

# macOS-specific filtering. APFS exposes many synthetic firmlinks and
# system snapshots; agent context wants user-visible volumes only.
_DISK_SKIP_FSTYPES_DARWIN = {"devfs", "autofs", "lifs", "nullfs"}
_DISK_SKIP_MOUNT_PREFIXES_DARWIN = (
    "/System/Volumes/VM",
    "/System/Volumes/Preboot",
    "/System/Volumes/Update",
    "/System/Volumes/xarts",
    "/System/Volumes/iSCPreboot",
    "/System/Volumes/Hardware",
    "/System/Volumes/Recovery",
    "/private/var/vm",
)


def _basename_to_disk(name: str) -> str:
    """Map a partition name back to its parent disk basename.
    sda1 -> sda; nvme0n1p3 -> nvme0n1; mmcblk0p2 -> mmcblk0."""
    if name.startswith(("nvme", "mmcblk")):
        idx = name.find("p")
        return name[:idx] if idx > 0 and name[idx + 1:].isdigit() else name
    return name.rstrip("0123456789") or name


def _read_block_devices() -> dict[str, dict[str, Any]]:
    """Return {basename: {model, size_bytes, rotational, bus, transport}} for
    physical block devices. Uses lsblk when present, /sys fallback otherwise.
    Unprivileged."""
    devices: dict[str, dict[str, Any]] = {}
    if shutil.which("lsblk"):
        try:
            res = subprocess.run(
                ["lsblk", "-J", "-b", "-d", "-o",
                 "NAME,TYPE,SIZE,ROTA,TRAN,MODEL,VENDOR"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if res.returncode == 0:
                for d in json.loads(res.stdout).get("blockdevices", []):
                    if d.get("type") != "disk":
                        continue
                    name = d["name"]
                    if name.startswith(_DISK_SKIP_DEV_PREFIXES):
                        continue
                    devices[name] = {
                        "model": (d.get("model") or "").strip() or None,
                        "vendor": (d.get("vendor") or "").strip() or None,
                        "size_bytes": int(d["size"]) if d.get("size") else None,
                        "rotational": bool(int(d["rota"])) if d.get("rota") is not None else None,
                        "transport": d.get("tran") or None,
                    }
        except (subprocess.SubprocessError, json.JSONDecodeError, ValueError, OSError):
            pass
    if devices:
        return devices
    # /sys fallback (no lsblk, or it errored)
    try:
        for entry in os.listdir("/sys/block"):
            if entry.startswith(_DISK_SKIP_DEV_PREFIXES):
                continue
            base = f"/sys/block/{entry}"
            size_sectors = _safe_read(f"{base}/size", limit=64)
            rota = _safe_read(f"{base}/queue/rotational", limit=4)
            model = _safe_read(f"{base}/device/model", limit=128)
            devices[entry] = {
                "model": model or None,
                "vendor": None,
                "size_bytes": int(size_sectors) * 512 if size_sectors.isdigit() else None,
                "rotational": rota == "1" if rota in ("0", "1") else None,
                "transport": None,
            }
    except OSError:
        pass
    return devices


def _read_mounts() -> list[tuple[str, str, str, list[str]]]:
    """Parse /proc/mounts -> [(source, mount_point, fstype, options)].
    Filters pseudo-fs, snap loops, and other agent-noise mounts."""
    out: list[tuple[str, str, str, list[str]]] = []
    raw = _safe_read("/proc/mounts", limit=65536)
    if not raw:
        return out
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        src, mnt, fstype, opts = parts[0], parts[1], parts[2], parts[3]
        if fstype in _DISK_SKIP_FSTYPES:
            continue
        if mnt.startswith(_DISK_SKIP_MOUNT_PREFIXES):
            continue
        # Octal-escape decode (\040 = space, \011 = tab) per fstab convention.
        for esc, ch in (("\\040", " "), ("\\011", "\t"), ("\\134", "\\")):
            mnt = mnt.replace(esc, ch)
            src = src.replace(esc, ch)
        out.append((src, mnt, fstype, opts.split(",")))
    return out


def _interesting_mount_opts(opts: list[str]) -> list[str]:
    """Keep only mount options the LLM would actually use."""
    keep = {"ro", "rw", "noatime", "relatime", "discard", "nodiratime",
            "sync", "async", "nodev", "nosuid", "noexec", "ssd", "compress"}
    return [o for o in opts if o in keep or o.startswith("compress=")]


def _gather_disk_facts_darwin() -> dict[str, Any]:
    """macOS host facts via `df -P -k`. Mount + fstype + usage only; no
    device sub-record (would require diskutil per-device parsing for marginal
    additional value). Same outer shape as the Linux path so the agent's
    consumer code does not branch."""
    out: dict[str, Any] = {"mounts": [], "unmounted_devices": []}
    if not shutil.which("df"):
        return out
    try:
        # -P = POSIX output (stable columns, no wrap); -k = 1K blocks;
        # -T <type> would filter, but BSD df spells filtering differently
        # from GNU df, so we filter in Python instead.
        res = subprocess.run(
            ["df", "-P", "-k"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if res.returncode != 0:
            return out
    except (subprocess.SubprocessError, OSError):
        return out

    # We need fstype too. `mount` is the reliable cross-mac source.
    fstype_by_mount: dict[str, str] = {}
    try:
        mres = subprocess.run(
            ["mount"], capture_output=True, text=True, timeout=5, check=False,
        )
        if mres.returncode == 0:
            # Lines look like:  /dev/disk3s1s1 on / (apfs, ...)
            import re as _re
            for line in mres.stdout.splitlines():
                m = _re.match(r"^(\S+) on (.+?) \(([^,)]+)", line)
                if m:
                    fstype_by_mount[m.group(2)] = m.group(3).strip()
    except (subprocess.SubprocessError, OSError):
        pass

    lines = res.stdout.splitlines()
    if not lines:
        return out
    # First line is the header; skip it.
    for line in lines[1:]:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        source, blocks_s, used_s, avail_s, _pct_s, mnt = parts
        try:
            blocks = int(blocks_s)
            used_k = int(used_s)
            avail_k = int(avail_s)
        except ValueError:
            continue

        fstype = fstype_by_mount.get(mnt, "")
        if fstype in _DISK_SKIP_FSTYPES_DARWIN:
            continue
        if mnt.startswith(_DISK_SKIP_MOUNT_PREFIXES_DARWIN):
            continue
        # devfs mounts surface as /dev — filter for safety even if `mount`
        # output was missed.
        if source == "devfs" or mnt == "/dev":
            continue

        total = blocks * 1024
        used = used_k * 1024
        free = avail_k * 1024
        pct = round(100.0 * used / total, 1) if total else None
        out["mounts"].append({
            "mount": mnt,
            "source": source,
            "fstype": fstype or None,
            "options": [],
            "total_gb": round(total / (1024 ** 3), 2),
            "free_gb": round(free / (1024 ** 3), 2),
            "used_gb": round(used / (1024 ** 3), 2),
            "percent_used": pct,
        })
    return out


def _gather_disk_facts(verbose: bool = False) -> dict[str, Any]:
    """Mount-first disk topology for the agent's host facts.

    Linux: full picture - inventory + sizes + usage + fstype + rotational
        + bus, plus optional 1s IO sample in verbose tier.
    macOS: mount + fstype + usage only (no device sub-record, no IO sample).
    Other platforms: empty stub; caller drops the key.

    Outer shape is identical across platforms so consumer code never branches.
    """
    out: dict[str, Any] = {"mounts": [], "unmounted_devices": []}
    system = platform.system()

    if system == "Darwin":
        return _gather_disk_facts_darwin()

    if system != "Linux":
        return out

    devices = _read_block_devices()
    mounts = _read_mounts()

    # Map each mount to its underlying physical disk basename, where derivable.
    mounted_disks: set[str] = set()
    for src, mnt, fstype, opts in mounts:
        dev_base: str | None = None
        if src.startswith("/dev/"):
            name = src[len("/dev/"):]
            dev_base = _basename_to_disk(name)
            if dev_base in devices:
                mounted_disks.add(dev_base)

        try:
            st = os.statvfs(mnt)
            total = st.f_blocks * st.f_frsize
            free = st.f_bavail * st.f_frsize
            used = total - (st.f_bfree * st.f_frsize)
            pct = round(100.0 * used / total, 1) if total else None
        except OSError:
            total = free = used = pct = None

        entry: dict[str, Any] = {
            "mount": mnt,
            "source": src,
            "fstype": fstype,
            "options": _interesting_mount_opts(opts),
            "total_gb": round(total / (1024 ** 3), 2) if total else None,
            "free_gb": round(free / (1024 ** 3), 2) if free else None,
            "used_gb": round(used / (1024 ** 3), 2) if used else None,
            "percent_used": pct,
        }
        if dev_base and dev_base in devices:
            d = devices[dev_base]
            entry["device"] = {
                "name": f"/dev/{dev_base}",
                "model": d.get("model"),
                "transport": d.get("transport"),
                "rotational": d.get("rotational"),
                "size_gb": round(d["size_bytes"] / (1024 ** 3), 2)
                            if d.get("size_bytes") else None,
            }
        out["mounts"].append(entry)

    for name, d in devices.items():
        if name in mounted_disks:
            continue
        out["unmounted_devices"].append({
            "name": f"/dev/{name}",
            "model": d.get("model"),
            "transport": d.get("transport"),
            "rotational": d.get("rotational"),
            "size_gb": round(d["size_bytes"] / (1024 ** 3), 2)
                        if d.get("size_bytes") else None,
        })

    if verbose:
        out["io_sample"] = _sample_disk_io(devices.keys())

    return out


def _read_diskstats() -> dict[str, tuple[int, int]]:
    """Parse /proc/diskstats -> {basename: (sectors_read, sectors_written)}.
    Sector size is 512 bytes per the kernel ABI, independent of physical
    block size."""
    snap: dict[str, tuple[int, int]] = {}
    raw = _safe_read("/proc/diskstats", limit=65536)
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 14:
            continue
        # fields: major minor name reads_completed reads_merged sectors_read
        # ms_reading writes_completed writes_merged sectors_written ...
        name = parts[2]
        if name.startswith(_DISK_SKIP_DEV_PREFIXES):
            continue
        try:
            snap[name] = (int(parts[5]), int(parts[9]))
        except ValueError:
            continue
    return snap


def _sample_disk_io(device_names) -> dict[str, dict[str, float]]:
    """1-second delta sample of /proc/diskstats. MB/s read/write per disk.
    Returns only entries for devices in `device_names` (the physical disks)."""
    import time as _t
    wanted = set(device_names)
    first = _read_diskstats()
    _t.sleep(1.0)
    second = _read_diskstats()
    result: dict[str, dict[str, float]] = {}
    for name in wanted:
        if name not in first or name not in second:
            continue
        sr1, sw1 = first[name]
        sr2, sw2 = second[name]
        # 512-byte sectors -> MB/s over 1s
        result[name] = {
            "read_mb_s": round((sr2 - sr1) * 512 / (1024 ** 2), 3),
            "write_mb_s": round((sw2 - sw1) * 512 / (1024 ** 2), 3),
        }
    return result


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
        # disk / filesystem inspection
        "lsblk", "findmnt", "df", "du", "blkid", "smartctl",
    ]
    facts["available_tools"] = _which_many(candidates)

    # Expanded but still privacy-conscious host facts. These are useful for
    # selecting the right commands without requiring a first probing turn.
    facts["hostname"] = socket.gethostname()
    facts["package_managers"] = _which_many([
        "apt", "apt-get", "dnf", "yum", "pacman", "zypper", "brew", "apk", "pkg"
    ])
    facts["init_tools"] = _which_many(["systemctl", "service", "launchctl", "rc-service"])
    facts["container_tools"] = _which_many(["docker", "podman", "kubectl"])
    facts["virtual_env"] = os.environ.get("VIRTUAL_ENV", "")

    try:
        usage = shutil.disk_usage(os.getcwd())
        facts["cwd_disk"] = {
            "total_gb": round(usage.total / (1024 ** 3), 2),
            "free_gb": round(usage.free / (1024 ** 3), 2),
        }
    except OSError:
        pass

    # Mount-first disk topology (Linux). Unprivileged, ~5-10ms cost.
    disk = _gather_disk_facts(verbose=False)
    if disk["mounts"] or disk["unmounted_devices"]:
        facts["disks"] = disk

    return facts


def gather_verbose_host_facts() -> dict[str, Any]:
    facts = gather_host_facts()
    facts["cpu_count"] = os.cpu_count()
    facts["path"] = os.environ.get("PATH", "")

    cgroup = _safe_read("/proc/1/cgroup", limit=4096).lower()
    facts["container_detected"] = any(needle in cgroup for needle in ("docker", "containerd", "kubepods"))

    network_tools = _which_many(["ip", "ifconfig", "netstat", "route"])
    facts["network_tools"] = network_tools

    # Live IO sample (1s window). Skipped on non-Linux or if default tier
    # already produced no disk facts.
    if "disks" in facts and platform.system() == "Linux":
        sample = _gather_disk_facts(verbose=True).get("io_sample")
        if sample:
            facts["disks"]["io_sample"] = sample

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


def _deny_real_argv(tokens: list[str]) -> list[str]:
    """Strip VAR=val assignments and wrapper commands; return the real argv."""
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if (not t.startswith("-") and "=" in t
                and t.split("=", 1)[0].isidentifier()):
            i += 1
            continue
        base = t.rsplit("/", 1)[-1]
        if base in _DENY_WRAPPERS:
            i += 1
            while i < len(tokens) and tokens[i].startswith("-"):
                consumes = tokens[i] in _DENY_OPTS_WITH_VALUE
                i += 1
                if consumes and i < len(tokens):
                    i += 1
            # timeout takes a mandatory positional DURATION before the command.
            if base == "timeout" and i < len(tokens):
                i += 1
            continue
        return tokens[i:]
    return []


def _deny_is_recursive(flags: list[str]) -> bool:
    return any(f == "--recursive" or _DENY_RECURSIVE_SHORT.match(f)
               for f in flags)


def _deny_argv(argv: list[str]) -> str | None:
    if not argv:
        return None
    cmd = argv[0].rsplit("/", 1)[-1]
    args = argv[1:]
    flags = [a for a in args if a.startswith("-") and a != "-"]
    positionals = [a for a in args if not a.startswith("-")]
    if cmd == "rm" and _deny_is_recursive(flags):
        if any(p in _DENY_ROOT_TARGETS for p in positionals):
            return "recursive delete targeting the filesystem root"
    if cmd == "chmod" and _deny_is_recursive(flags):
        # chmod's first positional is the mode; path operands follow.
        if any(p in _DENY_ROOT_TARGETS for p in positionals[1:]):
            return "recursive chmod on the filesystem root"
    if cmd == "mkfs" or cmd.startswith("mkfs."):
        return "filesystem format (mkfs)"
    if cmd == "dd" and any(_DENY_OF_DEV_RE.match(a) for a in args):
        return "dd writing directly to a raw block device"
    return None


def _deny_redirect(tokens: list[str]) -> str | None:
    for j, t in enumerate(tokens):
        m = _DENY_REDIR_RE.match(t)
        if not m:
            continue
        target = m.group(1) or (tokens[j + 1] if j + 1 < len(tokens) else "")
        if _DENY_BLOCKDEV_RE.match(target):
            return "redirection overwriting a raw block device"
    return None


def is_denied(cmd: str) -> str | None:
    """
    Return a human-readable reason if cmd matches a hard-deny rule, else None.
    Intent-based: each ;|&&-separated segment is tokenised and its real argv
    (past VAR=val assignments and sudo/env/timeout wrappers) is inspected, so
    legitimate non-root targets are permitted and trivial reformatting
    (flag order, extra spaces, /bin/rm) does not evade the rule.
    """
    if _FORKBOMB_RE.search(cmd):
        return "fork bomb"
    for seg in _DENY_SEG_RE.split(cmd):
        seg = seg.strip()
        if not seg:
            continue
        try:
            toks = shlex.split(seg)
        except ValueError:
            # Unbalanced quotes — fall back to a whitespace split.
            toks = seg.split()
        why = _deny_argv(_deny_real_argv(toks)) or _deny_redirect(toks)
        if why:
            return why
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
        except KeyboardInterrupt:
            # Ctrl-C here cancels just this command and returns to the REPL,
            # matching command-level interrupt handling. Session-quit stays
            # explicit: use 'q' or Ctrl-D.
            print(fail("\n[interrupted — command skipped]"))            
            return "skip", None
        if ans in ("y", "yes", ""):
            return "run", None
        if ans in ("n", "no"):
            return "skip", None
        if ans in ("q", "quit", "abort"):
            return "abort", None
        if ans in ("e", "edit"):
            # edit prompt plain — it's a user input position. The command is
            # pre-filled into the line buffer so it can be modified in place.
            try:
                new = _input_prefilled("edit> ", cmd).strip()
            except EOFError:
                return "abort", None
            except KeyboardInterrupt:
                # Cancel the edit, fall back to the y/n/e/q prompt.
                print(fail("\n[edit cancelled]"))                
                continue            
            if new:
                return "run", new


def _terminate_group(proc: subprocess.Popen) -> None:
    """
    Best-effort SIGTERM-then-SIGKILL of the command's entire process group.
    subprocess's own timeout only kills the immediate child (the shell), which
    leaves grandchildren (e.g. apt-get under /bin/sh) orphaned. Requires the
    process to have been started with start_new_session=True. POSIX only; on
    platforms without killpg this degrades to killing the direct child.
    """
    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    if not (killpg and getpgid):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return
    try:
        pgid = getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def execute(cmd: str) -> CmdResult:
    try:
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
    except Exception as e:                  # noqa: BLE001
        return CmdResult(cmd, 1, "", f"exec error: {e}")
    try:
        out, err = proc.communicate(timeout=COMMAND_TIMEOUT)
        return CmdResult(cmd, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        _terminate_group(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return CmdResult(cmd, 124, out or "", f"TIMEOUT after {COMMAND_TIMEOUT}s")
    except KeyboardInterrupt:
        # Kill the group before unwinding so Ctrl-C never orphans a command.
        _terminate_group(proc)
        raise
    except Exception as e:                  # noqa: BLE001
        _terminate_group(proc)
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
    input_tokens: int = 0       # non-cached input tokens sent THIS call
    output_tokens: int = 0      # tokens generated THIS call
    cache_read: int = 0         # input tokens read from prompt cache (Anthropic)
    cache_write: int = 0        # input tokens written to prompt cache (Anthropic)

    @property
    def context_tokens(self) -> int:
        # True context size. Anthropic reports input_tokens as the non-cached
        # portion only; cached tokens come back separately. OpenAI's
        # prompt_tokens is already the full count, so cache_read/cache_write
        # stay 0 there and this sum is a no-op.
        return self.input_tokens + self.cache_read + self.cache_write


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
                cache_read=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
                cache_write=getattr(
                    resp.usage, "cache_creation_input_tokens", 0) or 0,
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


def available_provider_names() -> list[str]:
    names: list[str] = []
    if os.environ.get("OPENAI_API_KEY"):
        names.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        names.append("anthropic")
    return names


def make_provider(name: str, model: str | None = None) -> Provider:
    normalized = name.strip().lower()
    if normalized in ("openai", "o"):
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set")
        return OpenAIProvider(model or DEFAULT_OPENAI_MODEL)
    if normalized in ("anthropic", "a", "claude"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY is not set")
        return AnthropicProvider(model or DEFAULT_ANTHROPIC_MODEL)
    raise ValueError("provider must be openai or anthropic")


def _select_model(provider_name: str, current: str) -> str | None:
    """
    Print a numbered list of valid models for `provider_name` and prompt for
    a choice. Returns the chosen model string, or None if the user cancelled
    (Ctrl-C / Ctrl-D / empty input) — caller should keep the current model.
    Accepts either the list number or an exact model name.
    """
    models = PROVIDER_MODELS.get(provider_name, ())
    if not models:
        print(err_bold(f"[no model list for provider {provider_name!r}]"))
        return None
    print(banner(f"Models for {provider_name}:"))
    for idx, name in enumerate(models, 1):
        marker = dim("  (current)") if name == current else ""
        ctx = CONTEXT_WINDOWS.get(name)
        ctx_str = dim(f"  [{ctx:,} ctx]") if ctx else ""
        print(f"  [{idx}] {name}{ctx_str}{marker}")
    try:
        ans = input_no_history(ask("Choice [number or name, blank to cancel]: ")).strip()
    except (EOFError, KeyboardInterrupt):
        print(fail("\n[model unchanged]"))
        return None
    if not ans:
        print(dim("[model unchanged]"))
        return None
    if ans.isdigit():
        i = int(ans)
        if 1 <= i <= len(models):
            return models[i - 1]
        print(err_bold(f"[invalid choice: {ans}]"))
        return None
    if ans in models:
        return ans
    print(err_bold(f"[not a valid model for {provider_name}: {ans!r}]"))
    return None


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
        - Each run_command call executes in its OWN fresh shell. Shell state
          does not persist between calls: a `cd`, an exported variable, or an
          activated venv in one command is gone by the next. To act within a
          directory, put the `cd` in the same command (`cd /var/log && ls`).
          Absolute paths also work and are fine for one-off commands.
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
    facts_verbose = False
    facts = gather_host_facts()
    system = build_system_prompt(facts)
    messages = provider.initial_messages(system)
    auto_approve = False
    show_tokens = False                  # /tokens on|off
    session_in = 0                       # cumulative input tokens this session
    session_out = 0                      # cumulative output tokens this session
    last_usage = Usage()                 # token usage from most recent call
    ctx_window = CONTEXT_WINDOWS.get(provider.model)   # may be None

    def fmt_tokens(u: Usage) -> str:
        ctx = u.context_tokens
        ctx_part = (
            f"{ctx}/{ctx_window} ({100 * ctx / ctx_window:.1f}%)"
            if ctx_window else f"{ctx} (window unknown)"
        )
        cache_part = (
            f" cache(r={u.cache_read} w={u.cache_write})"
            if (u.cache_read or u.cache_write) else ""
        )
        return dim(
            f"[tokens turn: in={u.input_tokens} out={u.output_tokens}{cache_part}  "
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
        except EOFError:            
            print()
            return
        except KeyboardInterrupt:
            # Ctrl-C at an idle prompt clears the line and re-prompts; it
            # never ends the session. Quit explicitly: /exit, /quit, Ctrl-D.
            print()
            continue
        if not user_in:
            continue

        # Add the typed line to history explicitly. input()'s implicit add is
        # disabled (init_readline) because it is unreliable with colored
        # prompts on stdlib readline; doing it here makes recall deterministic
        # across backends. Meta-commands are then dropped again so they do not
        # pollute Up-arrow recall of real conversational prompts.
        if _HAVE_READLINE:
            try:
                readline.add_history(user_in)
            except Exception:   # noqa: BLE001
                pass
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
            session_in = session_out = 0
            last_usage = Usage()
            print(dim("[conversation reset, token counters cleared]"))
            continue
        if user_in == "/info":
            ctx_used = last_usage.context_tokens
            ctx_str = (
                f"{ctx_used}/{ctx_window} ({100 * ctx_used / ctx_window:.1f}%)"
                if ctx_window and ctx_used
                else (str(ctx_window) if ctx_window else "unknown")
            )
            print(json.dumps({
                "provider": provider.name,
                "model": provider.model,
                "context_window": ctx_window,
                "session": {
                    "input_tokens": session_in,
                    "output_tokens": session_out,
                    "last_call_input_tokens": last_usage.input_tokens,
                    "last_call_output_tokens": last_usage.output_tokens,
                    "last_call_cache_read": last_usage.cache_read,
                    "last_call_cache_write": last_usage.cache_write,
                    "context_used_last_call": ctx_str,
                },
                "auto_approve": auto_approve,
                "show_tokens": show_tokens,
                "color": _color_enabled,
                "host": facts,
            }, indent=2))
            continue
        if user_in.startswith("/provider"):
            parts = user_in.split()
            if len(parts) == 1:
                print(json.dumps({
                    "current_provider": provider.name,
                    "current_model": provider.model,
                    "available_providers": available_provider_names(),
                }, indent=2))
                continue
            if len(parts) != 2 or parts[1].lower() not in ("openai", "anthropic", "o", "a", "claude"):
                print(dim("usage: /provider openai|anthropic"))
                continue
            try:
                provider = make_provider(parts[1])
            except ValueError as e:
                print(err_bold(f"[provider switch failed] {e}"))
                continue
            # Provider message formats differ, so reset conversation on switch.
            # Host facts and toggles are preserved.
            system = build_system_prompt(facts)
            messages = provider.initial_messages(system)
            session_in = session_out = 0
            last_usage = Usage()
            ctx_window = CONTEXT_WINDOWS.get(provider.model)
            print(dim(
                f"[provider switched to {provider.name}; model={provider.model}; "
                "conversation/token counters reset]"
            ))
            continue
        if user_in.startswith("/model"):
            parts = user_in.split(maxsplit=1)
            requested = parts[1].strip() if len(parts) == 2 else ""
            known = PROVIDER_MODELS.get(provider.name, ())
            prefixes = PROVIDER_MODEL_PREFIXES.get(provider.name, ())
            if requested and requested in known:
                chosen = requested
            elif requested and requested.startswith(prefixes):
                # Unlisted but the prefix matches this provider — likely a
                # newer release. Accept it, but warn it is unrecognised.
                print(warn(
                    f"[warning: {requested!r} is not a known {provider.name} "
                    "model — switching anyway; context-% may be unavailable]"
                ))
                chosen = requested
            else:
                # No arg, or a wrong-provider / nonsense name: show selector.
                if requested:
                    print(err_bold(
                        f"[not a valid {provider.name} model: {requested!r}]"
                    ))
                chosen = _select_model(provider.name, provider.model)
            if chosen is None or chosen == provider.model:
                continue
            provider.model = chosen
            ctx_window = CONTEXT_WINDOWS.get(provider.model)
            print(dim(f"[model switched to {provider.model}]"))
            continue
        if user_in.startswith("/facts"):
            parts = user_in.split()
            if len(parts) == 1:
                print(json.dumps(facts, indent=2))
                continue
            if len(parts) == 2 and parts[1] == "refresh":
                facts = gather_verbose_host_facts() if facts_verbose else gather_host_facts()
                system = build_system_prompt(facts)
                messages = provider.initial_messages(system)
                session_in = session_out = 0
                last_usage = Usage()
                print(dim("[host facts refreshed; conversation/token counters reset]"))
                continue
            if len(parts) == 3 and parts[1] == "verbose" and parts[2] in ("on", "off"):
                facts_verbose = parts[2] == "on"
                facts = gather_verbose_host_facts() if facts_verbose else gather_host_facts()
                system = build_system_prompt(facts)
                messages = provider.initial_messages(system)
                session_in = session_out = 0
                last_usage = Usage()
                print(dim(
                    f"[facts verbose = {facts_verbose}; host facts refreshed; "
                    "conversation/token counters reset]"
                ))
                continue
            print(dim("usage: /facts | /facts refresh | /facts verbose on|off"))
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
                print(fmt_tokens(last_usage))
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
            except KeyboardInterrupt:
                print(fail("\n[interrupted — request cancelled]"))
                # Same orphan-user-message rule as the api-error path below.
                if iteration == 0:
                    messages.pop()
                break
            except Exception as e:          # noqa: BLE001
                print(err_bold(f"[api error] {explain_api_error(e)}"))
                # Only pop on first iteration (orphan user msg);
                # on later iterations the conversation is in a valid state.
                if iteration == 0:
                    messages.pop()
                break
            iteration += 1

            # Accumulate token usage from this API call
            last_usage = turn.usage
            session_in += last_usage.input_tokens
            session_out += last_usage.output_tokens
            if show_tokens:
                print(fmt_tokens(last_usage))

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

                try:
                    result = execute(to_run)
                except KeyboardInterrupt:
                    print(fail("\n[interrupted — command cancelled]"))
                    results.append((tc.id, json.dumps(
                        {"error": "user interrupted command"})))
                    continue
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
