#!/usr/bin/env -S uv run --script
# /// script
# # Pin Python 3.13 so uv uses the same tested version on the Pi and Mac.
# # A wider range can keep an older cached interpreter. Update both bounds together.
# requires-python = ">=3.13,<3.14"
# dependencies = [
#     # Minimum SDK versions tested on the Pi and Mac. uv re-syncs when these change.
#     "openai>=3.19",
#     "anthropic>=1.8,<2.0",
#     # GNU readline keeps colored prompts and history working across POSIX hosts.
#     "gnureadline>=8.1; sys_platform != 'win32'",
# ]
# ///
"""
sys_agent.py — Minimal multi-provider shell agent (OpenAI / Anthropic /Deepseek).

Gathers host facts at startup, ships them to the selected model so command
syntax matches the actual environment, then executes model-proposed commands
locally only after explicit user approval. Uses native function/tool calling
on both providers (no fragile prose parsing).

Requires: uv  (https://astral.sh/uv) — installs openai + anthropic on first run
Run:      ./sys_agent.py            (after chmod +x)
          uv run sys_agent.py       (explicit form)

Env:      OPENAI_API_KEY            (one of these is required)
          ANTHROPIC_API_KEY
          DEEPSEEK_API_KEY
          SYS_PROVIDER               (skip prompt: openai|anthropic|deepseek)
          SYS_OPENAI_MODEL           (default gpt-6-luna)
          SYS_ANTHROPIC_MODEL        (default claude-haiku-5-5)
          SYS_DEEPSEEK_MODEL         (default deepseek-flash)
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

import os as _os
import sys as _sys

# Set the terminfo search path before importing readline. The bundled ncurses
# can miss Debian's /lib/terminfo, causing broken cursor movement and Backspace.
# Preserve any TERMINFO or TERMINFO_DIRS set by the user.
if (_sys.platform.startswith("linux")
        and "TERMINFO" not in _os.environ
        and "TERMINFO_DIRS" not in _os.environ):
    _terminfo_dirs = [d for d in ("/etc/terminfo", "/lib/terminfo",
                                  "/usr/lib/terminfo", "/usr/share/terminfo")
                      if _os.path.isdir(d)]
    if _terminfo_dirs:
        _os.environ["TERMINFO_DIRS"] = ":".join(_terminfo_dirs)

# Import readline before the API SDKs to avoid a crash on exit on macOS.
# Prefer GNU readline for colored prompts and history. Fall back to the
# standard library's readline where available.
try:
    import gnureadline as readline     # noqa: F401 — preferred backend
    _HAVE_READLINE = True
    _IS_LIBEDIT = False
except ImportError:
    try:
        import readline                # noqa: F401 — stdlib fallback
        _HAVE_READLINE = True
        # libedit can misdraw colored prompts on history recall. Detect it
        # so colored_input() can print the prompt prefix separately.
        _IS_LIBEDIT = bool(readline.__doc__ and "libedit" in readline.__doc__)
    except ImportError:
        _HAVE_READLINE = False
        _IS_LIBEDIT = False

import hashlib
import importlib.metadata
import json
import os
import platform
import queue
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import itertools
import atexit
import datetime
from dataclasses import dataclass, field
from typing import Any


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

# =============================================================================
# PROVIDER & MODEL CONFIG
# -----------------------------------------------------------------------------
# For a new model, update PROVIDER_MODELS, CONTEXT_WINDOWS, and any new
# prefix in PROVIDER_MODEL_PREFIXES. Check its reasoning settings below
# and record the probe results and listing rationale in docs/models.md.
# =============================================================================

# Startup defaults. Override in the environment or switch with /model.
# See docs/models.md for pricing, probe results, and the choice of defaults.
DEFAULT_OPENAI_MODEL = "gpt-6-luna"            # override: SYS_OPENAI_MODEL

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"   # override: SYS_ANTHROPIC_MODEL

# DeepSeek can update the model behind this alias without changing its name.
# Run the ds_* probes in tools/live_probe.py to check its behavior.
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"      # override: SYS_DEEPSEEK_MODEL

# Let the startup picker display model names without creating API clients.
DEFAULT_MODELS: dict[str, str] = {
    "openai": DEFAULT_OPENAI_MODEL,
    "anthropic": DEFAULT_ANTHROPIC_MODEL,
    "deepseek": DEFAULT_DEEPSEEK_MODEL,
}

# Use the OpenAI SDK with DeepSeek's endpoint.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Context limits in tokens, used for the percentage display. Unknown models
# show token counts only. Keep older entries for pins and manual switches.
CONTEXT_WINDOWS: dict[str, int] = {
    # OpenAI
    "gpt-5.4-mini":    400_000,
    "gpt-4o-mini":     128_000,
    "gpt-5.6-luna":  1_050_000,
    "gpt-5.6-terra": 1_050_000,
    "gpt-5.6-sol":   1_050_000,
    "gpt-6-astra":   1_050_000,
    "gpt-6-luna":    1_050_000,
    "gpt-6-sol":     1_050_000,
    "gpt-6.1-sol":   1_050_000,
    "gpt-6.1-terra": 1_050_000,
    "gpt-6.2-astra": 1_050_000,
    "gpt-6.2-sol":   1_050_000,
    # Anthropic
    "claude-haiku-4-5-20251001": 200_000,
    "claude-haiku-5-5":        1_000_000,
    "claude-sonnet-5-5":       1_000_000,
    "claude-sonnet-5":         1_000_000,
    "claude-sonnet-4-6":       1_000_000,
    "claude-opus-4-7":         1_000_000,
    "claude-opus-4-8":         1_000_000,
    "claude-opus-5":           1_000_000,
    "claude-opus-5-5":         1_000_000,
    "claude-fable-5":          1_000_000,
    "claude-fable-5-1":        1_000_000,
    "claude-fable-5-2":        1_000_000,
    # DeepSeek
    "deepseek-flash":          1_000_000,
    "deepseek-v4-flash":       1_000_000,  # Old name kept for existing pins.
    "deepseek-v4-pro":         1_000_000,
}

# Models suggested by /model. Unlisted names with a recognized prefix still
# work, with a warning.
PROVIDER_MODELS: dict[str, tuple[str, ...]] = {
    "openai": (
        "gpt-6-luna",
        "gpt-6.1-terra",
        "gpt-6-sol",
        "gpt-6.2-sol",
        "gpt-5.4-mini",
    ),
    "anthropic": (
        "claude-haiku-5-5",
        "claude-sonnet-5-5",
        "claude-opus-5-5",
    ),
    "deepseek": (
        "deepseek-flash",
        "deepseek-v4-pro",
    ),
}

# Use these prefixes to check unlisted model names against the active provider.
PROVIDER_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-", "o1", "o3", "o4", "chatgpt-"),
    "anthropic": ("claude-",),
    "deepseek": ("deepseek-",),
}

# Reasoning levels each model accepts with function tools, lowest first,
# copied from its 400 for an unsupported value (no model accepts "minimal").
# Keyed by exact name because levels vary within a generation (gpt-6-sol takes
# "none", gpt-6.1-sol does not). A model with no entry gets no reasoning field:
# required for gpt-4o-mini, which 400s on one, and the safe default for an
# unprobed model, though /thinking off then cannot stop it reasoning.
_OPENAI_EFFORTS_FULL = ("none", "low", "medium", "high", "xhigh", "max")
_OPENAI_EFFORTS_NO_NONE = ("low", "medium", "high", "xhigh", "max")
_OPENAI_REASONING_EFFORTS: dict[str, tuple[str, ...]] = {
    "gpt-6-luna":    _OPENAI_EFFORTS_FULL,
    "gpt-6.1-terra": _OPENAI_EFFORTS_FULL,
    "gpt-6-sol":     _OPENAI_EFFORTS_FULL,
    "gpt-5.6-luna":  _OPENAI_EFFORTS_FULL,
    "gpt-5.6-terra": _OPENAI_EFFORTS_FULL,
    "gpt-5.6-sol":   _OPENAI_EFFORTS_FULL,
    "gpt-5.4-mini":  ("none", "low", "medium", "high", "xhigh"),
    "gpt-6.1-sol":   _OPENAI_EFFORTS_NO_NONE,
    "gpt-6.2-sol":   _OPENAI_EFFORTS_NO_NONE,
    "gpt-6-astra":   _OPENAI_EFFORTS_NO_NONE,
    "gpt-6.2-astra": _OPENAI_EFFORTS_NO_NONE,
}

# =============================================================================
# RUNTIME TUNING
# =============================================================================

# Use the first env file found unless SYS_ENV_FILE selects one. Shell
# variables take priority. Expand paths and variable references on lookup.
def _default_env_candidates() -> list[str]:
    xdg = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return [
        ".env",                              # CWD (dotenv convention)
        os.path.join(xdg, "sys_agent", ".env"),
        "~/.sys_agent.env",                  # home dotfile fallback
    ]

# Limit stdout and stderr sent to the model, in characters per stream.
# Keep both ends because errors often appear at the end.
OUTPUT_MAX_CHARS = 8000

# Limit how much output execute() buffers per stream while a command runs.
# OUTPUT_MAX_CHARS applies only after the command ends, so without this an
# endless writer (`yes`, unbounded journalctl) grew to gigabytes on the Pi.
# Past the limit the process group is stopped and the command reports 125.
OUTPUT_CAPTURE_MAX_BYTES = 4 * 1024 * 1024

# Command timeout in seconds, overridable with SYS_COMMAND_TIMEOUT.
# Allow time for package upgrades on slower hosts. Stop the whole process
# group on timeout so helpers such as dpkg do not keep running on their own.
COMMAND_TIMEOUT = 120                          # override: SYS_COMMAND_TIMEOUT

# Anthropic requires max_tokens; pick something generous for tool dialogs.
ANTHROPIC_MAX_TOKENS = 4096

# Anthropic thinking defaults to off to save billed output tokens on routine
# commands. Set the startup state with SYS_THINKING or toggle with /thinking.
# Adaptive models use effort to guide how much they think. Legacy models use
# a token budget and need a beta header to keep thinking across tool calls.
ANTHROPIC_THINKING_DEFAULT = False             # override: SYS_THINKING

# effort: low|medium|high|xhigh|max. Steers adaptive thinking depth; "high" is
# the API default. Ignored on the "enabled" (Haiku/legacy) path.
_VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")
ANTHROPIC_THINKING_EFFORT_DEFAULT = "high"     # override: SYS_THINKING_EFFORT

# budget_tokens for the legacy "enabled" path only (Haiku 4.5 / older).
ANTHROPIC_THINKING_BUDGET = 4000               # override: SYS_THINKING_BUDGET

# Allow room for both reasoning and the reply on the same turn. 32K stays under
# every current model's output limit and bills only for tokens produced.
ANTHROPIC_THINKING_MAX_TOKENS = 32000          # override: SYS_THINKING_MAX_TOKENS

# Interleaved-thinking beta header — needed only on the "enabled" path. It is
# deprecated/ignored on adaptive models, so it is NOT sent in adaptive mode.
ANTHROPIC_INTERLEAVED_BETA = "interleaved-thinking-2025-05-14"

# These models use a fixed thinking budget. Unknown models default to adaptive.
def _thinking_mode(model: str) -> str:
    """'enabled' for Haiku 4.5 / older Claude models, else 'adaptive'."""
    if "haiku-4-5" in model:
        return "enabled"
    return "adaptive"

# These models think by default, so /thinking off needs an explicit setting.
# Sonnet 5.5 only supports "between_tools", which can still produce short
# thinking blocks between tool calls. Omit effort because xhigh and max
# reject these settings. Verify additions against the Models API.
_THINKING_OFF_PAYLOAD: dict[str, dict[str, str]] = {
    "claude-sonnet-5": {"type": "disabled"},
    "claude-opus-5": {"type": "disabled"},
    "claude-haiku-5-5": {"type": "disabled"},
    "claude-sonnet-5-5": {"type": "between_tools"},
}

def _thinking_off_payload(model: str) -> dict[str, str] | None:
    return _THINKING_OFF_PAYLOAD.get(model)

# These models always think: {"type":"disabled"} 400s, so they must never be
# in _THINKING_OFF_PAYLOAD (a test keeps the two disjoint). With /thinking off,
# send no effort (server default) but keep the thinking-turn token limit, since
# the 4096 cap could cut off a hard turn, and stream, since the SDK rejects
# that limit on non-streaming requests.
_THINKING_ALWAYS_ON_MODELS = frozenset({
    "claude-opus-5-5", "claude-fable-5", "claude-fable-5-1",
    "claude-fable-5-2",
})

def _thinking_always_on(model: str) -> bool:
    return model in _THINKING_ALWAYS_ON_MODELS

def _thinking_unverified(model: str) -> bool:
    """True for an Anthropic model this file knows nothing about: one reached
    through the "claude-" prefix fallback (/model or a SYS_ANTHROPIC_MODEL
    pin) that has no CONTEXT_WINDOWS entry. Every model whose thinking
    behavior has been checked has one.

    Such a model is usually a new release, and every release since Sonnet 5
    thinks when no thinking field is sent; some cannot stop. Handled as a
    plain no-thinking model it got the 4096-token cap and the non-streaming
    call, so if it did think, a hard turn was truncated with nothing to say
    why (the Fable 5.2 gap the model monitor flagged before v1.29.0). With
    thinking off it therefore gets the always-on shape: no thinking field (no
    off form is known, and a wrong one 400s), the thinking-turn cap, and
    streaming. That shape is valid whether or not the model thinks; the cap
    bills only for tokens produced. Legacy-named models (_thinking_mode
    "enabled") are excluded: that generation does not think by default.
    """
    return model not in CONTEXT_WINDOWS and _thinking_mode(model) == "adaptive"

def _thinking_headroom(model: str) -> bool:
    """Whether a thinking-off turn must still be sized and streamed for
    thinking: the model cannot stop (always-on) or might not (unverified)."""
    return _thinking_always_on(model) or _thinking_unverified(model)

def _openai_effort(model: str, thinking: bool, effort: str) -> str | None:
    """reasoning.effort to send for an OpenAI model, or None for no field.

    /thinking off sends the model's lowest grade: "none" where accepted,
    else "low" (6.1/6.2-sol and the Astras always reason). /thinking on sends
    the session effort, clamped down to the highest grade the model accepts
    (max -> xhigh on gpt-5.4-mini), since sending an unlisted grade 400s."""
    levels = _OPENAI_REASONING_EFFORTS.get(model)
    if not levels:
        return None
    if not thinking:
        return levels[0]
    if effort in levels:
        return effort
    rank = _VALID_EFFORTS.index(effort) if effort in _VALID_EFFORTS else 0
    below = [lvl for lvl in levels
             if lvl in _VALID_EFFORTS and _VALID_EFFORTS.index(lvl) <= rank]
    return below[-1] if below else levels[-1]

def _openai_always_reasons(model: str) -> bool:
    """True for an OpenAI model whose lowest grade still reasons."""
    levels = _OPENAI_REASONING_EFFORTS.get(model)
    return bool(levels) and levels[0] != "none"

# SDK retries use backoff and honor Retry-After for temporary API errors.
API_MAX_RETRIES = 5

# Remind users to review proposed commands. Set SYS_DISCLAIMER=off to hide it.
SHOW_DISCLAIMER = True                         # override: SYS_DISCLAIMER

# Check for an update at startup and display a notice when one is available.
# Notify only: an agent that runs shell commands must not replace its own code.
UPDATE_CHECK = True                            # override: SYS_UPDATE_CHECK
UPDATE_REPO_API = "https://api.github.com/repos/mikeoc61/sys_agent"
UPDATE_BRANCH = "main"
UPDATE_HTTP_TIMEOUT = 3.0     # per request; the check makes at most two
UPDATE_BANNER_WAIT = 1.0      # max extra startup delay waiting for the result
DISCLAIMER = (
    "model-proposed commands can be confidently wrong — LLMs hallucinate "
    "flags, paths, and facts, and may lag current software versions. The "
    "approval prompt is your review point: read each command before running "
    "(/auto on disables it)."
)

# Update this list whenever run_repl gains or changes a slash command.
META_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help",           "Show this command list and current toggle states"),
    ("/info",           "Provider, model, session token usage, host facts"),
    ("/version",        "sys_agent version, GitHub main status, runtime versions"),
    ("/reset",          "Clear conversation history and token counters"),
    ("/auto on|off",    "Skip the per-command approval prompt (deny list still applies)"),
    ("/thinking on|off", "Toggle extended thinking — Anthropic/DeepSeek/OpenAI reasoning models (next turn)"),
    ("/effort [level]", "Thinking depth low|medium|high|xhigh|max — Anthropic adaptive models and OpenAI reasoning models (all 5; gpt-5.4-mini caps at xhigh), DeepSeek (low|high|max); needs /thinking on"),
    ("/tokens on|off",  "Toggle the per-turn token-usage line"),
    ("/color on|off",   "Toggle ANSI color output"),
    ("/audit [on|off]", "Show or toggle the command audit log"),
    ("/history [N|all]", "Review recent command history from the audit log (paged)"),
    ("/consult [--fresh]", "Second opinion from the other providers (executes nothing); --fresh drops prior history"),
    ("/provider [name]", "Show or switch provider: openai|anthropic|deepseek"),
    ("/model [name]",    "Switch model for the active provider; no arg lists choices"),
    ("/facts",           "Print current host facts"),
    ("/facts refresh",   "Re-probe host facts and rebuild system prompt"),
    ("/facts verbose on|off", "Toggle expanded host fact collection"),
    ("/exit, /quit",    "End the session"),
)

# Keep the startup list short. /help lists every command.
BANNER_META_COMMANDS: tuple[str, ...] = (
    "/info", "/model", "/provider", "/thinking", "/consult", "/auto", "/exit",
)

# Block these dangerous patterns even after approval. This is a backstop, not
# the security boundary: the approval prompt is the gate. The checks inspect
# arguments, so "rm -rf /home/x" can still reach the approval prompt
# while "rm -rf /" and "sudo rm -rf /*" are blocked.

# Check for fork bombs before splitting on shell separators.
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
    MAGENTA   = "\033[35m"
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
def host_tag(t: str) -> str:   return _wrap(t, _ANSI.BOLD, _ANSI.MAGENTA)
def agent_tag(t: str) -> str:  return _wrap(t, _ANSI.BOLD, _ANSI.GREEN)


def visible(text: str, newlines: bool = True) -> tuple[str, int]:
    """(display form, hidden-character count) for model- or log-supplied text
    shown where the user makes a decision.

    The approval prompt is the security gate, so what it shows must be what
    runs. Printed raw, a proposed command can carry terminal control
    sequences (ESC[2K + CR erases the line it was just printed on) or
    Unicode format characters (bidi overrides, zero-width) that make the
    displayed text differ from the string handed to the shell. Every control
    (Cc), format (Cf) and line/paragraph-separator character is rendered as
    a visible \\xNN / \\uNNNN escape instead. Tab stays, and newline stays
    when `newlines` is true: a multi-line command (heredoc) is legitimate and
    displays honestly. The count lets the caller say that escapes are
    present, since a typed literal backslash-x looks the same.
    """
    out: list[str] = []
    hidden = 0
    for ch in text:
        if ch == "\t" or (ch == "\n" and newlines):
            out.append(ch)
        elif unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp"):
            hidden += 1
            code = ord(ch)
            out.append(f"\\x{code:02x}" if code <= 0xff
                       else f"\\u{code:04x}" if code <= 0xffff
                       else f"\\U{code:08x}")
        else:
            out.append(ch)
    return "".join(out), hidden


def short_model(model: str) -> str:
    """Compact model label for the agent> line: strip a trailing -YYYYMMDD pin
    (claude-haiku-4-5-20251001 -> claude-haiku-4-5). Current dateless IDs pass
    through unchanged. The startup banner keeps the full pinned ID; this only
    keeps the per-answer label tight."""
    return re.sub(r"-\d{8}$", "", model)


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


# input() reads one line at a time. A delimiter lets users submit pasted
# lines together on both GNU readline and libedit.
PASTE_DELIM = '"""'


def read_paste_block(first: str) -> str:
    """
    Collect a multi-line message opened by a line starting with PASTE_DELIM.
    Text after the opening delimiter is the first line; a line ending with the
    delimiter closes the block, and its text before the delimiter is kept, so
    both `\"\"\"` on its own line and `\"\"\"text ... text\"\"\"` work. Ctrl-D
    closes the block too (end of input, like `cat`). Ctrl-C propagates so the
    caller can discard the partial block — never ends the session.

    Known limit: pasted content with a line ending in the delimiter (Python
    docstrings) closes the block early.
    """
    body = first[len(PASTE_DELIM):]
    if body.rstrip().endswith(PASTE_DELIM) and body.strip() != "":
        return body.rstrip()[:-len(PASTE_DELIM)].strip()
    lines = [body] if body.strip() else []
    while True:
        try:
            line = colored_input(dim("... "))
        except EOFError:
            print()
            break
        if line.rstrip().endswith(PASTE_DELIM):
            tail = line.rstrip()[:-len(PASTE_DELIM)]
            if tail.strip():
                lines.append(tail)
            break
        lines.append(line)
    return "\n".join(lines).strip("\n").rstrip()


# -----------------------------------------------------------------------------
# readline (input history + line editing)
# -----------------------------------------------------------------------------

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
    # GNU readline can skip history entries for colored prompts. Add them
    # ourselves, and use input_no_history() for prompts like command approval.
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
# Progress while waiting for the model
# -----------------------------------------------------------------------------
#
# Show a spinner and elapsed time during slow model calls. Use one static
# line for redirected output or TERM=dumb. Set SYS_PROGRESS=off to disable.
# Join the spinner thread before clearing its line so writes stay ordered.
SHOW_PROGRESS = True                           # override: SYS_PROGRESS
_SPINNER_INTERVAL = 0.1                 # seconds between frames
_SPINNER_DELAY = 1.5                    # suppress the spinner for ops faster than this


def _spinner_frames() -> tuple[str, ...]:
    """ASCII spinner — renders in every terminal/font. (Braille glyphs look
    nicer but are font-dependent and came out invisible in some terminals,
    leaving only the elapsed counter visible.)"""
    return ("|", "/", "-", "\\")


class Activity:
    """Context manager that shows the model is working during provider.chat().

    TTY: a background daemon thread paints `<spinner> <label>… <n>s` on one
    line until __exit__, which stops + joins it and erases the line (CR+EL).
    Non-TTY / TERM=dumb: one static `[<label>…]` line, no control characters.
    SYS_PROGRESS=off: nothing. The thread writes only while the caller is
    blocked in chat(), and is joined before the clear, so no write races.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._painted = False
        self._animate = (
            SHOW_PROGRESS
            and sys.stdout.isatty()
            and os.environ.get("TERM", "") != "dumb"
        )

    def __enter__(self) -> "Activity":
        if not SHOW_PROGRESS:
            return self
        if not self._animate:
            print(dim(f"[{self.label}…]"), flush=True)
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        start = time.monotonic()
        # Skip the indicator for calls that finish quickly.
        if self._stop.wait(_SPINNER_DELAY):
            return
        self._painted = True
        for frame in itertools.cycle(_spinner_frames()):
            if self._stop.is_set():
                break
            elapsed = int(time.monotonic() - start)
            sys.stdout.write("\r" + dim(f"{frame} {self.label}… {elapsed}s "))
            sys.stdout.flush()
            self._stop.wait(_SPINNER_INTERVAL)

    def __exit__(self, *exc: Any) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._painted:
            # Clear the spinner line before printing the reply.
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


# -----------------------------------------------------------------------------
# Command audit log
# -----------------------------------------------------------------------------
#
# Record proposed commands, the approval decision, and any exit code.
# SYS_AUDIT_LOG sets the path or disables logging with "off" or an empty value.
# SYS_AUDIT_BODY=on includes capped stdout and stderr, which may carry secrets.
# Each record is one JSON line. The file grows until manually rotated or cleared.
AUDIT_LOG_DEFAULT = "~/.config/sys_agent/audit.log"
_AUDIT_OFF_VALUES = frozenset(("", "off", "0", "false", "none"))

# Module-level state, set by init_audit() and the /audit REPL command.
_audit_path: str | None = None
_audit_body: bool = False


def _resolve_audit_path(raw: str) -> str | None:
    """Expand `raw` to an audit-log path and ensure its directory exists.
    Returns None when `raw` requests disable, or on a dir-create failure
    (printed once, dimmed). Non-fatal — a failure just disables auditing."""
    if raw.strip().lower() in _AUDIT_OFF_VALUES:
        return None
    path = os.path.expanduser(os.path.expandvars(raw.strip()))
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except OSError as e:
        print(dim(f"[audit: cannot create log dir: {e}; auditing off]"))
        return None
    return path


def init_audit() -> None:
    """Resolve audit-log state from the environment at startup. Must run after
    the env file is loaded so SYS_AUDIT_LOG/SYS_AUDIT_BODY from it are honored."""
    global _audit_path, _audit_body
    _audit_path = _resolve_audit_path(
        os.environ.get("SYS_AUDIT_LOG", AUDIT_LOG_DEFAULT)
    )
    _audit_body = os.environ.get("SYS_AUDIT_BODY", "off").strip().lower() == "on"


def write_audit(record: dict[str, Any]) -> None:
    """Append one JSON record as a single line. No-op when auditing is off.
    On a write failure, auditing is disabled (rather than re-erroring on every
    subsequent command) after a one-line notice."""
    global _audit_path
    if _audit_path is None:
        return
    try:
        with open(_audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(dim(f"[audit: write failed ({e}); auditing disabled]"))
        _audit_path = None


def _fmt_local_time(ts: str) -> tuple[str, str]:
    """Convert a UTC 'Z' audit timestamp to (date, time) in the host's local
    zone — humans reading a history want wall-clock, not UTC. Falls back to a
    naive string split if the timestamp doesn't parse."""
    try:
        dt = (datetime.datetime
              .strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
              .replace(tzinfo=datetime.timezone.utc)
              .astimezone())
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        d, _, t = ts.partition("T")
        return d, t.rstrip("Z")


def render_history(path: str, limit: int | None) -> str | None:
    """Read the audit-log JSONL at `path` and render a numbered, human-readable
    history in host-local time. `limit` keeps only the most recent N records;
    None shows all. Day-change separators disambiguate multi-session logs.
    Returns the rendered text, or None when there is no history to show.

    Effective command is the edited form when the user rewrote it; deny/skip/
    abort are tagged and carry no exit code (they never ran)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            records: list[dict[str, Any]] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue            # tolerate a partially-written tail line
    except OSError:
        return None
    if not records:
        return None

    total = len(records)
    shown = records if limit is None else records[-limit:]
    span = "all" if (limit is None or limit >= total) else f"last {len(shown)}"
    width = len(str(len(shown)))

    lines = [dim(f"# audit history — {span} of {total} records  (host-local time)")]
    last_date: str | None = None
    for i, r in enumerate(shown, 1):
        date, tm = _fmt_local_time(r.get("ts", ""))
        if date != last_date:
            lines.append(dim(f"── {date} ──"))
            last_date = date
        action = r.get("action", "?")
        cmd = r.get("edited_command") or r.get("command", "")
        why = r.get("explanation", "")
        rc = r.get("returncode")

        tag = ""
        if action == "edit":
            tag = warn(" [edited]")
        elif action == "skip":
            tag = dim(" [skipped]")
        elif action == "deny":
            tag = fail(f" [denied: {r.get('reason', '')}]")
        elif action == "abort":
            tag = fail(" [aborted]")
        rc_str = fail(f" (exit {rc})") if rc not in (None, 0) else ""

        line = (f"{i:>{width}}.  {dim(tm)}  "
                f"{cmd_label(visible(cmd)[0])}{tag}{rc_str}")
        if why:
            line += f"  {dim('— ' + visible(why, newlines=False)[0])}"
        lines.append(line)
    return "\n".join(lines)


def page_text(text: str) -> None:
    """Display `text` through a pager when interactive; print plainly otherwise.
    Honors $PAGER; defaults to `less -RFX` (color-safe, auto-quits when it fits
    one screen), then `more`. Falls back to a plain print when there is no
    pager, stdout is not a tty, or the pager fails to spawn."""
    if not text.endswith("\n"):
        text += "\n"
    pager_argv: list[str] | None = None
    if sys.stdout.isatty():
        env_pager = os.environ.get("PAGER", "").strip()
        if env_pager:
            pager_argv = shlex.split(env_pager)
        elif shutil.which("less"):
            pager_argv = ["less", "-RFX"]
        elif shutil.which("more"):
            pager_argv = ["more"]
    if not pager_argv:
        print(text, end="")
        return
    try:
        proc = subprocess.Popen(pager_argv, stdin=subprocess.PIPE, text=True)
    except OSError:
        print(text, end="")
        return
    try:
        proc.communicate(text)
    except (BrokenPipeError, KeyboardInterrupt):
        # Close the pager and return to the prompt on q or Ctrl-C.
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


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


def _probe(argv: list[str], timeout: float = 5) -> str | None:
    """stdout of a read-only fact probe, or None when the tool is not on PATH,
    fails to spawn, times out, or exits non-zero — every fact probe treats
    those alike and falls back or omits the fact. stdin is detached so a probe
    can never read the user's terminal; undecodable bytes are replaced rather
    than raising, as in _safe_read. Callers that read the exit status itself
    as the answer (VM detection, sudo) run subprocess directly."""
    if not shutil.which(argv[0]):
        return None
    try:
        res = subprocess.run(argv, stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, errors="replace",
                             timeout=timeout, check=False)
    except (subprocess.SubprocessError, OSError):
        return None
    return res.stdout if res.returncode == 0 else None


# Check both sysstat directories for binary saDD or saYYYYMMDD files.
# The sarDD files are text reports.
_SAR_DATA_DIRS = ("/var/log/sysstat", "/var/log/sa")
_SAR_DATA_FILE = re.compile(r"^sa(\d{2}|\d{8})$")


def _gather_sar_history() -> dict[str, Any]:
    """Whether `sar` is installed AND has collected history. Installing the
    package puts `sar` on PATH but, on Debian/Pi OS, collection stays off
    until ENABLED="true" in /etc/default/sysstat; a bare `which` therefore
    cannot tell a usable `sar -u` from "Cannot open /var/log/sysstat/saDD".
    Empty when sar is absent (available_tools already says so); otherwise
    `daily_files` of 0 means installed but never collected. Stat-only, no
    subprocess; the data itself is live and left to commands."""
    if not shutil.which("sar"):
        return {}
    for d in _SAR_DATA_DIRS:
        try:
            n = sum(1 for f in os.listdir(d) if _SAR_DATA_FILE.match(f))
        except OSError:
            continue
        return {"data_dir": d, "daily_files": n}
    return {"daily_files": 0}


# Memory-policy sources (module constants so tests can point them at fixtures).
_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_CGROUPS = "/proc/cgroups"
_PSI_MEMORY = "/proc/pressure/memory"
_SWAPPINESS = "/proc/sys/vm/swappiness"
_PROC_SWAPS = "/proc/swaps"
_ZSWAP_ENABLED = "/sys/module/zswap/parameters/enabled"


def _cgroup_memory_v1_enabled() -> bool:
    """`enabled` column of the memory row in /proc/cgroups. v1/hybrid only:
    on v2 the file is not authoritative (6.12 Pi kernels omit the memory row
    entirely when CONFIG_MEMCG_V1 is off, though v2 memcg is live)."""
    for line in _safe_read(_PROC_CGROUPS, limit=4096).splitlines():
        f = line.split()
        if len(f) >= 4 and f[0] == "memory":
            return f[3] == "1"
    return False


def _gather_memory_policy_linux() -> dict[str, Any]:
    """Effective kernel memory policy — what the kernel resolved, not what
    config requests. Config and effect diverge in practice: Raspberry Pi
    firmware prepends `cgroup_disable=memory`, leaving systemd MemoryHigh=/
    MemoryMax= inert while `systemctl show` still prints them; and on Debian
    /etc/sysctl.conf (linked as 99-sysctl.conf) loads after most
    /etc/sysctl.d files and silently overrides them. Raw /proc/cmdline is
    deliberately not read: it holds both requests (pibot carries
    cgroup_disable= AND cgroup_enable=) plus root=PARTUUID identifiers.

      cgroup         v2 | hybrid | v1
      cgroup_memory  memory controller active (v2: root cgroup.controllers)
      psi            /proc/pressure/memory readable (Pi kernels need psi=1)
      swappiness     live vm.swappiness
      swap_kinds     sorted kinds from /proc/swaps: file | partition | zram
      zswap          compressed swap cache on; absent when not built

    Keys attached only when resolved. Pressure averages, memory.events
    counters, and swap used are live and stay commands."""
    mp: dict[str, Any] = {}
    root = _CGROUP_ROOT
    if os.path.exists(os.path.join(root, "cgroup.controllers")):
        mp["cgroup"] = "v2"
        mp["cgroup_memory"] = "memory" in _safe_read(
            os.path.join(root, "cgroup.controllers")).split()
    elif os.path.exists(os.path.join(root, "unified", "cgroup.controllers")):
        mp["cgroup"] = "hybrid"
        mp["cgroup_memory"] = _cgroup_memory_v1_enabled()
    elif os.path.isdir(root) and _safe_read(_PROC_CGROUPS, limit=64):
        mp["cgroup"] = "v1"
        mp["cgroup_memory"] = _cgroup_memory_v1_enabled()
    # With PSI disabled, _safe_read() returns an empty string.
    mp["psi"] = _safe_read(_PSI_MEMORY, limit=256).startswith("some")
    sw = _safe_read(_SWAPPINESS, limit=16)
    if sw.isdigit():
        mp["swappiness"] = int(sw)
    kinds: set[str] = set()
    for line in _safe_read(_PROC_SWAPS, limit=4096).splitlines()[1:]:
        f = line.split()
        if len(f) >= 2:
            # zram registers as type "partition"; the device name tells.
            kinds.add("zram" if f[0].startswith("/dev/zram") else f[1])
    if kinds:
        mp["swap_kinds"] = sorted(kinds)
    zs = _safe_read(_ZSWAP_ENABLED, limit=8)
    if zs in ("Y", "N"):
        mp["zswap"] = zs == "Y"
    return mp


# Hide virtual devices and filesystems from the disk inventory.
# Keep the device prefixes in sync with disk_smart.NO_SMART_PREFIXES.
_DISK_SKIP_DEV_PREFIXES = ("loop", "ram", "zram", "dm-", "sr", "fd")
_DISK_SKIP_FSTYPES = {
    "proc", "sysfs", "devtmpfs", "devpts", "tmpfs", "cgroup", "cgroup2",
    "pstore", "bpf", "tracefs", "debugfs", "configfs", "fusectl",
    "securityfs", "mqueue", "hugetlbfs", "autofs", "rpc_pipefs",
    "binfmt_misc", "nsfs", "overlay", "squashfs", "ramfs",
}
# Hide snap, container, and temporary system mounts from the inventory.
_DISK_SKIP_MOUNT_PREFIXES = (
    "/snap/", "/var/lib/docker/", "/var/lib/containers/",
    "/run/", "/sys/", "/proc/", "/dev/",
)

# Hide Time Machine snapshot mounts under /Volumes/.timemachine. They repeat
# the backup disk's totals and can crowd out other volumes. Keep the backup
# volumes themselves in the list.
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
    "/Volumes/.timemachine",
)


# Map the USB enclosure's vendor:product ID to the smartctl -d option it
# needs. The system provides the ID, but the option depends on the bridge
# chip. Keep this in sync with disk_smart.py's KNOWN_BRIDGE_HINTS.
_USB_BRIDGE_SMART_HINTS = {
    "04e8:4001": "sntasmedia",   # Samsung Portable SSD T7 (NVMe / ASMedia)
    "174c:55aa": "sntasmedia",   # ASMedia ASM1153/1351 generic
    "174c:1153": "sntasmedia",
    "152d:0578": "sntjmicron",   # JMicron JMS578 and kin
    "152d:0583": "sntjmicron",
    "152d:9561": "sntjmicron",
}


def _usb_bridge_ids(basename: str) -> tuple[str, str] | None:
    """Resolve a USB block device's bridge (vendor_id, product_id) as 4-hex
    lowercase strings, or None when the device is not USB-attached or the IDs
    can't be read. Prefers `udevadm` (authoritative; it walks the USB device
    tree); falls back to ascending sysfs for idVendor/idProduct. Unprivileged,
    Linux-only. Returns None for non-USB transports so callers can attach the
    `usb_ids` key only when it actually means a USB bridge."""
    out = _probe(["udevadm", "info", "--query=property", "-n", f"/dev/{basename}"])
    if out is not None:
        props: dict[str, str] = {}
        for line in out.splitlines():
            k, _, v = line.partition("=")
            if v:
                props[k] = v
        if props.get("ID_BUS") == "usb":
            vid = props.get("ID_VENDOR_ID")
            pid = props.get("ID_MODEL_ID")
            if vid and pid:
                return vid.strip().lower(), pid.strip().lower()
        return None                     # resolved, and it is not USB
    # If udevadm is unavailable, look for idVendor and idProduct in sysfs
    # by walking up from the block device.
    try:
        path = os.path.realpath(f"/sys/block/{basename}/device")
    except OSError:
        return None
    for _ in range(8):
        vid = _safe_read(os.path.join(path, "idVendor"), limit=16)
        pid = _safe_read(os.path.join(path, "idProduct"), limit=16)
        if vid and pid:
            return vid.strip().lower(), pid.strip().lower()
        parent = os.path.dirname(path)
        if parent == path or not parent.startswith("/sys"):
            break
        path = parent
    return None


def _augment_usb_smart_hints(devices: dict[str, dict[str, Any]]) -> None:
    """For USB-attached disks, attach `usb_ids` (the bridge vendor:product) and,
    when the bridge is recognized, `smartctl_device_type` (the `-d` token that
    reaches SMART through it). Mutates each device dict in place; no-op for
    non-USB transports and off Linux. Probes only devices whose transport is
    `usb` or unknown (the sysfs-fallback case), so native SATA/NVMe disks cost
    nothing extra."""
    if platform.system() != "Linux":
        return
    for name, d in devices.items():
        if d.get("transport") not in ("usb", None):
            continue
        ids = _usb_bridge_ids(name)
        if not ids:
            continue
        vid, pid = ids
        d["usb_ids"] = {"vendor": vid, "product": pid}
        hint = _USB_BRIDGE_SMART_HINTS.get(f"{vid}:{pid}")
        if hint:
            d["smartctl_device_type"] = hint


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
    out = _probe(["lsblk", "-J", "-b", "-d", "-o",
                  "NAME,TYPE,SIZE,ROTA,TRAN,MODEL,VENDOR"])
    if out is not None:
        try:
            for d in json.loads(out).get("blockdevices", []):
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
        except ValueError:              # JSONDecodeError is a ValueError
            pass
    if devices:
        _augment_usb_smart_hints(devices)
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
    _augment_usb_smart_hints(devices)
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
    # Use fixed columns and 1 KiB blocks. Filter in Python because macOS
    # and Linux use different df options for filesystem type.
    df_out = _probe(["df", "-P", "-k"])
    if df_out is None:
        return out

    # Read the filesystem type from mount, e.g. "/dev/disk3s1s1 on / (apfs, ...)".
    fstype_by_mount: dict[str, str] = {}
    for line in (_probe(["mount"]) or "").splitlines():
        m = re.match(r"^(\S+) on (.+?) \(([^,)]+)", line)
        if m:
            fstype_by_mount[m.group(2)] = m.group(3).strip()

    lines = df_out.splitlines()
    if not lines:
        return out
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
        # Hide /dev even if mount could not identify its filesystem type.
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


def _device_subrecord(basename: str, d: dict[str, Any]) -> dict[str, Any]:
    """Shape a physical-device dict into the agent-facing sub-record. Single
    source of truth shared by the mounted and unmounted paths, so the two
    cannot drift. Attaches `usb_ids` and `smartctl_device_type` only when the
    USB-bridge probe resolved them (see _augment_usb_smart_hints)."""
    rec: dict[str, Any] = {
        "name": f"/dev/{basename}",
        "model": d.get("model"),
        "transport": d.get("transport"),
        "rotational": d.get("rotational"),
        "size_gb": round(d["size_bytes"] / (1024 ** 3), 2)
                    if d.get("size_bytes") else None,
    }
    if d.get("usb_ids"):
        rec["usb_ids"] = d["usb_ids"]
    if d.get("smartctl_device_type"):
        rec["smartctl_device_type"] = d["smartctl_device_type"]
    return rec


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
            entry["device"] = _device_subrecord(dev_base, devices[dev_base])
        out["mounts"].append(entry)

    for name, d in devices.items():
        if name in mounted_disks:
            continue
        out["unmounted_devices"].append(_device_subrecord(name, d))

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
    wanted = set(device_names)
    first = _read_diskstats()
    time.sleep(1.0)
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


# -----------------------------------------------------------------------------
# Hardware identity (static platform facts)
# -----------------------------------------------------------------------------
# Gather static hardware facts at startup so the model knows which metrics
# it can ask for, such as battery status on a laptop or IPMI on a server.
# These checks need no elevated permissions.

# Map SMBIOS enclosure codes to broad hardware types such as laptop or server.
# The codes come from SMBIOS System Enclosure (type 3).
_DMI_CHASSIS_CLASS: dict[int, str] = {
    3: "desktop", 4: "desktop", 5: "desktop", 6: "desktop", 7: "desktop",
    13: "desktop", 24: "desktop", 34: "desktop", 35: "desktop",
    8: "laptop", 9: "laptop", 10: "laptop", 11: "laptop", 14: "laptop",
    30: "laptop", 31: "laptop", 32: "laptop",
    17: "server", 23: "server", 25: "server", 28: "server", 29: "server",
}

# Ignore common placeholder values in hardware identity fields.
_DMI_JUNK_VALUES = frozenset((
    "to be filled by o.e.m.", "system product name", "system manufacturer",
    "system version", "default string", "none", "unknown", "not specified",
    "not applicable", "oem",
))

# Check these product names for a VM when systemd-detect-virt is unavailable.
_DMI_VM_NEEDLES = (
    "kvm", "qemu", "vmware", "virtualbox", "virtual machine", "hvm domu",
    "xen", "amazon ec2", "parallels", "bochs",
)

# Use DMI vendor and product names to identify the cloud provider.
# Query the metadata service on demand for details such as region or instance ID.
_DMI_CLOUD_NEEDLES = (
    ("amazon ec2", "aws"),
    ("google compute engine", "gcp"),
    ("oraclecloud", "oci"),
    ("digitalocean", "digitalocean"),
)
# Azure and local Hyper-V share a vendor name. Identify Azure by its chassis tag.
_DMI_AZURE_ASSET_TAG = "7783-7084-3265-9085-8269-3286-77"


def _dmi_read(field: str) -> str:
    """Read a /sys/class/dmi/id field, filtering vendor placeholder junk.
    Returns '' when absent, unreadable, or junk. Unprivileged fields only
    (sys_vendor/product_name/chassis_type are world-readable; the root-only
    serial/UUID fields are never touched)."""
    val = _safe_read(f"/sys/class/dmi/id/{field}", limit=256).strip()
    return "" if val.lower() in _DMI_JUNK_VALUES else val


def _container_detected() -> bool:
    """Best-effort container detection. /.dockerenv is the cheap positive;
    the /proc/1/cgroup grep covers containerd/k8s/lxc but is known to
    false-negative on pure cgroup-v2 hosts — acceptable for a prior."""
    if os.path.exists("/.dockerenv"):
        return True
    cgroup = _safe_read("/proc/1/cgroup", limit=4096).lower()
    return any(n in cgroup for n in ("docker", "containerd", "kubepods", "lxc"))


def _vm_detected_linux(dmi_text: str) -> bool:
    """VM detection: systemd-detect-virt --vm when present (authoritative;
    exit 0 means a VM was detected), else DMI needles against the combined
    vendor+product text — QEMU puts "QEMU" in sys_vendor, VMware in
    product_name, so neither field alone covers both."""
    if shutil.which("systemd-detect-virt"):
        try:
            res = subprocess.run(
                ["systemd-detect-virt", "--vm"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            return res.returncode == 0
        except (subprocess.SubprocessError, OSError):
            pass
    t = dmi_text.lower()
    return any(n in t for n in _DMI_VM_NEEDLES)


def _cloud_provider_linux(dmi_text: str) -> str:
    """Cloud platform from DMI, or '' if not a recognized cloud.

    A classification (aws|gcp|azure|oci|digitalocean), not identity: it reveals
    the platform, never the instance. Sources are world-readable DMI fields VM
    detection also reads. Vendor/product needles are checked first (catches
    Nitro EC2, where sys_vendor is "Amazon EC2", plus GCP/OCI/DigitalOcean).
    Two fallbacks follow for hosts those needles miss:
      - Xen-generation EC2 (t2/m4/...) reports sys_vendor "Xen" /
        product_name "HVM domU" — no "amazon" in either — but stamps the BIOS
        (bios_version e.g. "4.11.amazon", bios_vendor occasionally "Amazon").
      - Azure sets the generic "Microsoft Corporation" vendor (shared with
        on-prem Hyper-V), disambiguated by its fixed chassis_asset_tag.
    Both fallback fields are world-readable and non-identifying. No IMDS call —
    instance metadata (region, instance-id, IAM role) stays a live, on-demand
    query, never a startup prior. /sys/hypervisor/uuid would also identify
    Xen-EC2 but is a UUID-class file, so it is deliberately not read."""
    t = dmi_text.lower()
    for needle, label in _DMI_CLOUD_NEEDLES:
        if needle in t:
            return label
    bios = f"{_dmi_read('bios_vendor')} {_dmi_read('bios_version')}".lower()
    if "amazon" in bios:
        return "aws"
    if _dmi_read("chassis_asset_tag").startswith(_DMI_AZURE_ASSET_TAG):
        return "azure"
    return ""


def _memory_gb_linux() -> float | None:
    """Total RAM from /proc/meminfo MemTotal, in GB (1 decimal)."""
    for line in _safe_read("/proc/meminfo", limit=4096).splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return round(int(parts[1]) * 1024 / (1024 ** 3), 1)
    return None


def _swap_gb_linux() -> float | None:
    """Configured swap from /proc/meminfo SwapTotal, GB (1 decimal). None when
    the field is absent; 0.0 is preserved (swap configured off) because it is a
    meaningful signal — on a low-RAM host, no swap means memory pressure ends in
    OOM-kills rather than swapping."""
    for line in _safe_read("/proc/meminfo", limit=4096).splitlines():
        if line.startswith("SwapTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return round(int(parts[1]) * 1024 / (1024 ** 3), 1)
    return None


def _sysctl_value(name: str) -> str:
    """`sysctl -n <name>` (macOS path); '' on any failure."""
    return (_probe(["sysctl", "-n", name]) or "").strip()


def _mac_has_battery() -> bool:
    """Laptop detection on macOS: an internal battery in `pmset -g batt`.
    Model identifiers stopped encoding the form factor on Apple Silicon
    ('Mac15,6' is an M3 MacBook Pro, 'Mac14,12' is a Mac Studio), so battery
    presence is the robust laptop signal, not the model-string prefix."""
    return "InternalBattery" in (_probe(["pmset", "-g", "batt"]) or "")


def _hwmon_sensors_linux() -> list[str]:
    """Names of /sys/class/hwmon devices (e.g. cpu_thermal, rpi_volt, nvme).
    Derived from sysfs, not curated — tells the model which sensor channels
    exist on this host without maintaining a map. Default-tier: a handful of
    short strings, and the strongest "this telemetry exists" signal — the
    channel names sit in context even if prompt guidance changes."""
    names: list[str] = []
    try:
        entries = sorted(os.listdir("/sys/class/hwmon"))
    except OSError:
        return names
    for e in entries:
        name = _safe_read(f"/sys/class/hwmon/{e}/name", limit=64)
        if name:
            names.append(name)
    return names


def _gather_hardware_facts() -> dict[str, Any]:
    """Static hardware identity: vendor, model, platform_class, memory_gb,
    swap_gb, and cloud (plus cpu_model on macOS, where the model identifier
    alone is opaque). `cloud` is the DMI-derived platform label
    (aws|gcp|azure|oci|...) when the host is a recognized cloud instance,
    absent otherwise.

    Fields are attached only when resolved, so key-presence is meaningful.
    `platform_class` is one of container|vm|sbc|laptop|desktop|server,
    resolved in that precedence order (a Pi inside Docker is a container
    first). Sources by platform:

      Linux SBC   /proc/device-tree/model  (authoritative on Pi/BeagleBone)
      Linux x86   /sys/class/dmi/id/{sys_vendor,product_name,chassis_type}
      macOS       sysctl hw.model / machdep.cpu.brand_string / hw.memsize,
                  pmset -g batt for the laptop/desktop split

    NOTE: the Darwin path is fixture-reasoned, not yet Mac-verified (same
    status the runtime snapshot shipped with). Validate with:
        sysctl -n hw.model machdep.cpu.brand_string hw.memsize
        pmset -g batt
    """
    hw: dict[str, Any] = {}
    system = platform.system()
    if system == "Linux":
        # Device-tree model is NUL-terminated in sysfs; strip it.
        dt_model = _safe_read(
            "/proc/device-tree/model", limit=256).rstrip("\x00").strip()
        dmi_vendor = _dmi_read("sys_vendor")
        dmi_product = _dmi_read("product_name")
        if dt_model:
            hw["model"] = dt_model
        elif dmi_product:
            hw["model"] = dmi_product
        if dmi_vendor:
            hw["vendor"] = dmi_vendor
        mem = _memory_gb_linux()
        if mem:
            hw["memory_gb"] = mem
        swap = _swap_gb_linux()
        if swap is not None:
            hw["swap_gb"] = swap
        cloud = _cloud_provider_linux(f"{dmi_vendor} {dmi_product}")
        if cloud:
            hw["cloud"] = cloud
        if _container_detected():
            hw["platform_class"] = "container"
        elif _vm_detected_linux(f"{dmi_vendor} {dmi_product}"):
            hw["platform_class"] = "vm"
        elif dt_model:
            hw["platform_class"] = "sbc"
        else:
            chassis = _safe_read("/sys/class/dmi/id/chassis_type", limit=8)
            cls = (_DMI_CHASSIS_CLASS.get(int(chassis))
                   if chassis.isdigit() else None)
            if cls:
                hw["platform_class"] = cls
    elif system == "Darwin":
        model = _sysctl_value("hw.model")
        if model:
            hw["vendor"] = "Apple"
            hw["model"] = model
            hw["platform_class"] = ("laptop" if _mac_has_battery()
                                    else "desktop")
        cpu = _sysctl_value("machdep.cpu.brand_string")
        if cpu:
            hw["cpu_model"] = cpu
        memsize = _sysctl_value("hw.memsize")
        if memsize.isdigit():
            hw["memory_gb"] = round(int(memsize) / (1024 ** 3), 1)
    return hw


# -----------------------------------------------------------------------------
# Runtime snapshot (running services + top processes by memory)
# -----------------------------------------------------------------------------
# Give the model real service and process names at startup. These are
# point-in-time facts. Use /facts refresh to gather them again.

# Cap the service list so it cannot crowd out the rest of the model's prompt.
RUNTIME_MAX_SERVICES = 80
# Top-N processes by aggregate resident memory. Override with SYS_TOP_PROCESSES.
RUNTIME_TOP_PROCESSES = 10                     # override: SYS_TOP_PROCESSES
# Hide Apple system jobs and open GUI apps from the macOS service list.
# GUI apps can still appear in the list of processes using the most memory.
_DARWIN_SERVICE_SKIP_PREFIXES = ("com.apple.", "application.")


def _top_processes_by_rss_linux(limit: int) -> list[dict[str, Any]]:
    """Top processes by aggregate RSS, read straight from /proc.

    Deliberately avoids `ps`: /proc is the one interface that behaves the same
    across distros, init systems, and userlands (procps vs busybox `ps` differ
    on `-o` format support), which matters for a multi-distro tool.

    Privacy: only basename(argv[0]) is kept; the rest of the command line —
    which can carry tokens/passwords in argv — is discarded before anything is
    recorded, so it never reaches the remote API. Processes sharing a name
    (worker pools) are aggregated into one entry with summed RSS and an
    instance count, which is the right signal for "what is using memory".
    """
    try:
        page = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        page = 4096
    agg: dict[str, list[int]] = {}      # name -> [summed_rss_bytes, instances]
    try:
        pids = [e for e in os.listdir("/proc") if e.isdigit()]
    except OSError:
        return []
    for pid in pids:
        # statm fields: size resident shared text lib data dt (in pages).
        statm = _safe_read(f"/proc/{pid}/statm", limit=128)
        if not statm:
            continue
        try:
            resident_pages = int(statm.split()[1])
        except (IndexError, ValueError):
            continue
        if resident_pages == 0:
            continue                    # kernel threads, fully-swapped, etc.
        # Read raw bytes to split arguments on NUL. Kernel threads have no cmdline.
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read(4096)
        except OSError:
            continue                    # process vanished mid-scan
        if not raw:
            continue                    # kernel thread (no userspace argv)
        argv0 = raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
        name = os.path.basename(argv0) or argv0
        if not name:
            continue
        slot = agg.setdefault(name, [0, 0])
        slot[0] += resident_pages * page
        slot[1] += 1
    ranked = sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)[:limit]
    out: list[dict[str, Any]] = []
    for name, (rss_bytes, instances) in ranked:
        entry: dict[str, Any] = {
            "name": name,
            "rss_mb": round(rss_bytes / (1024 ** 2), 1),
        }
        if instances > 1:
            entry["count"] = instances
        out.append(entry)
    return out


def _systemd_services(state: str, limit: int, user: bool = False) -> list[str]:
    """systemd service units in `state` ("running" | "failed") via
    `systemctl list-units`, when present. Script-stable flags only, matching
    the guidance the prompt gives the model.

    `user=False` queries the system manager; `user=True` queries the per-user
    manager (`systemctl --user`), which is where services a user runs for
    themselves live — e.g. a `bitcoind.service` started with `systemctl
    --user`, invisible to the system-scope list. The user query needs a
    reachable user D-Bus session (present when sys_agent runs in the invoking
    user's login session; absent under sudo, cron, or a user without linger).

    Failed units are queried at SYSTEM scope only: high-signal for turn-one
    triage (a unit named here is a failure the model would otherwise spend a
    discovery pass finding), while user-scope failures are rarer and carry the
    same bus caveat, so they are left to an explicit `systemctl --user
    --failed` when a user unit is in question.

    Returns [] on non-systemd hosts (OpenRC, runit, sysvinit, ...): enumerating
    every init system is out of scope, and on those hosts `top_processes` still
    carries the naming signal through the process basename. Also [] when the
    relevant bus is unreachable (container, no user bus), since list-units
    exits non-zero there — and, for failed units, on a healthy host.
    """
    argv = ["systemctl", *(["--user"] if user else []), "list-units",
            "--type=service", f"--state={state}",
            "--no-legend", "--plain", "--no-pager"]
    units: list[str] = []
    for line in (_probe(argv) or "").splitlines():
        parts = line.split()
        # --plain --no-legend => "UNIT LOAD ACTIVE SUB DESCRIPTION"; take UNIT.
        if parts and parts[0].endswith(".service"):
            units.append(parts[0])
            if len(units) >= limit:
                break
    return units


def _top_processes_by_rss_darwin(limit: int) -> list[dict[str, Any]]:
    """Top processes by aggregate RSS on macOS, via BSD `ps`.

    macOS has no /proc, so this uses `ps -axo rss=,comm=`. `comm` is the full
    executable PATH with no arguments (unlike `command`/`args`), which buys two
    things: argv — and any secrets in it — never leaves the host, and macOS app
    paths containing spaces survive, since basename() handles
    `/Applications/Some App.app/Contents/MacOS/Some App` correctly whereas
    whitespace-splitting `command` would not. `rss` is reported in 1024-byte
    units. Same aggregation as Linux: programs sharing a name are summed with an
    instance count.

    NOTE: depends on macOS `ps -o comm=` emitting the full executable path
    rather than the truncated accounting name (`ucomm`, MAXCOMLEN=16). If a
    given macOS build truncates here, long names would clip the same way the
    Linux /proc/comm cap does — validate on the Mac (see module header).
    """
    out = _probe(["ps", "-axo", "rss=,comm="])
    if out is None:
        return []
    agg: dict[str, list[int]] = {}      # name -> [summed_rss_kb, instances]
    for line in out.splitlines():
        # Split once to preserve spaces in application paths.
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        rss_s, comm = parts
        try:
            rss_kb = int(rss_s)
        except ValueError:
            continue                    # header/garbage line
        name = os.path.basename(comm.strip()) or comm.strip()
        if not name:
            continue
        slot = agg.setdefault(name, [0, 0])
        slot[0] += rss_kb
        slot[1] += 1
    ranked = sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)[:limit]
    out: list[dict[str, Any]] = []
    for name, (rss_kb, instances) in ranked:
        entry: dict[str, Any] = {"name": name, "rss_mb": round(rss_kb / 1024, 1)}
        if instances > 1:
            entry["count"] = instances
        out.append(entry)
    return out


def _running_services_darwin(limit: int) -> list[str]:
    """Running launchd jobs on macOS via `launchctl list` — the analog to the
    Linux systemd-unit list.

    Filtered to currently-running jobs (numeric PID column) with Apple-owned
    labels removed (`_DARWIN_SERVICE_SKIP_PREFIXES`). Scope is the invoking
    user's launchd domain (unprivileged), mirroring the Linux probe running as
    the invoking user. Label only — no paths or args. Note the naming-disambig
    payoff is softer than the systemd case: launchd labels are reverse-DNS
    (`homebrew.mxcl.postgresql`) and rarely diverge from what you'd guess the
    way `openclaw` -> `openclaw-gateway.service` did.
    """
    out = _probe(["launchctl", "list"])
    if out is None:
        return []
    labels: list[str] = []
    # Output is TAB-separated: "PID\tStatus\tLabel", with a header row.
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            parts = line.split()
        if len(parts) < 3:
            continue
        pid, label = parts[0], parts[-1]
        if not pid.isdigit():
            continue                    # not currently running ("-")
        if label.startswith(_DARWIN_SERVICE_SKIP_PREFIXES):
            continue                    # Apple system noise
        labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def _gather_runtime_facts_darwin() -> dict[str, Any]:
    """macOS runtime snapshot: launchd jobs + top processes by RSS, mirroring
    the Linux probe's shape and aggregation.

    NOT YET behaviorally verified on macOS from the Linux dev environment — the
    parsers are fixture-tested, but two facts only the Mac can confirm: that
    `ps -axo rss=,comm=` prints the full exec path (not the truncated `ucomm`),
    and the exact `launchctl list` column format. Validate before trusting:
        ps -axo rss=,comm= | sort -rn | head
        launchctl list | head
    """
    return {
        "running_services": _running_services_darwin(RUNTIME_MAX_SERVICES),
        "top_processes": _top_processes_by_rss_darwin(RUNTIME_TOP_PROCESSES),
    }


def _gather_runtime_facts() -> dict[str, Any]:
    """Startup runtime snapshot: {'running_services': [...], 'top_processes':
    [...]}. Linux implemented; macOS stubbed; other platforms empty. Outer
    shape is stable so `gather_host_facts` attaches only the non-empty keys."""
    system = platform.system()
    if system == "Linux":
        return {
            "running_services": _systemd_services("running", RUNTIME_MAX_SERVICES),
            "running_services_user": _systemd_services(
                "running", RUNTIME_MAX_SERVICES, user=True),
            "failed_services": _systemd_services("failed", RUNTIME_MAX_SERVICES),
            "top_processes": _top_processes_by_rss_linux(RUNTIME_TOP_PROCESSES),
        }
    if system == "Darwin":
        return _gather_runtime_facts_darwin()
    return {}


def _boot_epoch() -> float | None:
    """Epoch seconds of last boot, or None. Linux: now - /proc/uptime[0].
    macOS: the 'sec' field of `sysctl -n kern.boottime`. Unprivileged on both."""
    system = platform.system()
    if system == "Linux":
        data = _safe_read("/proc/uptime", limit=64).split()
        if data:
            try:
                return time.time() - float(data[0])
            except ValueError:
                return None
        return None
    if system == "Darwin":
        m = re.search(r"sec\s*=\s*(\d+)", _sysctl_value("kern.boottime"))
        if m:
            return float(m.group(1))
    return None


def _gather_network_interfaces() -> dict[str, str]:
    """Stable interface inventory: name -> kind (wifi|ethernet|thunderbolt|
    bridge|bluetooth|other). Names and roles are stable per host; IPs, MACs,
    SSIDs, and link state are deliberately excluded (volatile and/or
    identifying) — live network state is a command away. Linux keeps only
    device-backed interfaces (a `device` entry under /sys/class/net/<if>),
    which naturally drops lo/veth/docker/bridge noise; wireless is detected
    by the `wireless` subdir. Darwin parses `networksetup
    -listallhardwareports` (stock, unprivileged) and classifies from the
    Hardware Port label — "Thunderbolt Ethernet" is a NIC, so the ethernet
    test precedes the thunderbolt test. Returns {} on any failure."""
    system = platform.system()
    out: dict[str, str] = {}
    if system == "Linux":
        base = "/sys/class/net"
        try:
            names = sorted(os.listdir(base))
        except OSError:
            return {}
        for name in names:
            path = os.path.join(base, name)
            if not os.path.exists(os.path.join(path, "device")):
                continue
            wireless = os.path.isdir(os.path.join(path, "wireless"))
            out[name] = "wifi" if wireless else "ethernet"
        return out
    if system == "Darwin":
        ports = _probe(["networksetup", "-listallhardwareports"])
        if ports is None:
            return {}
        port = ""
        for line in ports.splitlines():
            if line.startswith("Hardware Port:"):
                port = line.partition(":")[2].strip()
            elif line.startswith("Device:") and port:
                dev = line.partition(":")[2].strip()
                label = port.lower()
                if "wi-fi" in label or "airport" in label:
                    kind = "wifi"
                elif "bluetooth" in label:
                    kind = "bluetooth"
                elif "bridge" in label:
                    kind = "bridge"
                elif "ethernet" in label:
                    kind = "ethernet"
                elif "thunderbolt" in label:
                    kind = "thunderbolt"
                else:
                    kind = "other"
                if dev:
                    out[dev] = kind
                port = ""
        return out
    return {}


def _gather_time_facts() -> dict[str, Any]:
    """Stable temporal orientation: timezone + last-boot time + uptime_hours.
    Unprivileged and cross-platform (boot time from /proc/uptime on Linux,
    kern.boottime on macOS). Keys attached only when resolved. Lets the model
    interpret log timestamps and gauge reboot recency on turn one without a
    probe; current wall-clock is still a live `date`/`uptime` away."""
    out: dict[str, Any] = {}
    now = time.localtime()
    tzname = now.tm_zone or (
        time.tzname[1] if now.tm_isdst > 0 else time.tzname[0])
    off = now.tm_gmtoff
    if tzname:
        if off is not None:
            sign = "+" if off >= 0 else "-"
            hh, mm = divmod(abs(off) // 60, 60)
            out["timezone"] = f"{tzname} (UTC{sign}{hh:02d}:{mm:02d})"
        else:
            out["timezone"] = tzname
    boot = _boot_epoch()
    if boot is not None:
        out["boot_time"] = datetime.datetime.fromtimestamp(
            boot).isoformat(timespec="seconds")
        out["uptime_hours"] = round((time.time() - boot) / 3600, 1)
    return out


# Check whether approved commands can use sudo. They run in a new session
# without a terminal, so sudo cannot prompt or use the user's cached password.
# The probe runs under the same conditions (new session, no stdin), which also
# catches a sudoers requiretty. Only running "sudo -n true" answers reliably:
# "sudo -n -v" fails on a passwordless host, and "sudo -n -l true" succeeds
# where a password is required. The probe can write to the system auth log.
# "password_required" also covers users with no sudo access.
SUDO_PROBE_TIMEOUT = 3.0


def _sudo_mode() -> str | None:
    """How sudo behaves for an approved command, or None when the probe was
    inconclusive (timeout, spawn error) so nothing is asserted either way."""
    if os.geteuid() == 0:
        return "running_as_root"
    if not shutil.which("sudo"):
        return "not_installed"
    try:
        proc = subprocess.run(
            ["sudo", "-n", "true"], stdin=subprocess.DEVNULL,
            capture_output=True, timeout=SUDO_PROBE_TIMEOUT,
            start_new_session=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return "passwordless" if proc.returncode == 0 else "password_required"


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
        # cloud / infra automation
        "aws", "gcloud", "az", "terraform", "ansible",
        # remote access / crypto
        "ssh", "scp", "openssl",
        # python ecosystem
        "python3", "pip", "pip3", "uv", "poetry", "pipx",
        # disk / filesystem inspection
        "lsblk", "findmnt", "df", "du", "blkid", "smartctl",
        # performance history (sysstat package)
        "sar", "sadf", "iostat", "mpstat",
        # hardware / power / sensor telemetry
        "vcgencmd", "sensors", "upower", "dmidecode", "ipmitool",
    ]
    facts["available_tools"] = _which_many(candidates)

    facts["hostname"] = socket.gethostname()
    facts["package_managers"] = _which_many([
        "apt", "apt-get", "dnf", "yum", "pacman", "zypper", "brew", "apk", "pkg"
    ])
    facts["init_tools"] = _which_many(["systemctl", "service", "launchctl", "rc-service"])
    facts["container_tools"] = _which_many(["docker", "podman", "kubectl"])
    facts["virtual_env"] = os.environ.get("VIRTUAL_ENV", "")

    # Omit sudo status if the probe could not complete.
    sudo = _sudo_mode()
    if sudo:
        facts["sudo"] = sudo

    # Include interface names and types. Leave addresses and link status out.
    interfaces = _gather_network_interfaces()
    if interfaces:
        facts["network_interfaces"] = interfaces

    # Timezone and boot details help the model interpret logs on its first turn.
    facts.update(_gather_time_facts())

    hardware = _gather_hardware_facts()
    if hardware:
        facts["hardware"] = hardware

    # Read sensor names from Linux hwmon, such as cpu_thermal or pwmfan.
    if platform.system() == "Linux":
        hwmon = _hwmon_sensors_linux()
        if hwmon:
            facts["hwmon_sensors"] = hwmon
        memory_policy = _gather_memory_policy_linux()
        if memory_policy:
            facts["memory_policy"] = memory_policy

    sar = _gather_sar_history()
    if sar:
        facts["sar_history"] = sar

    try:
        usage = shutil.disk_usage(os.getcwd())
        facts["cwd_disk"] = {
            "total_gb": round(usage.total / (1024 ** 3), 2),
            "free_gb": round(usage.free / (1024 ** 3), 2),
        }
    except OSError:
        pass

    disk = _gather_disk_facts(verbose=False)
    if disk["mounts"] or disk["unmounted_devices"]:
        facts["disks"] = disk

    # Include only the service and process lists we could fill.
    runtime = _gather_runtime_facts()
    if runtime.get("running_services"):
        facts["running_services"] = runtime["running_services"]
    if runtime.get("running_services_user"):
        facts["running_services_user"] = runtime["running_services_user"]
    if runtime.get("failed_services"):
        facts["failed_services"] = runtime["failed_services"]
    if runtime.get("top_processes"):
        facts["top_processes"] = runtime["top_processes"]

    return facts


def gather_verbose_host_facts() -> dict[str, Any]:
    facts = gather_host_facts()
    facts["cpu_count"] = os.cpu_count()
    facts["path"] = os.environ.get("PATH", "")

    facts["container_detected"] = _container_detected()

    network_tools = _which_many(["ip", "ifconfig", "netstat", "route"])
    facts["network_tools"] = network_tools

    # Sample disk I/O for one second when Linux disk facts are available.
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

# Chat Completions shape (nested "function"); DeepSeek only.
CHAT_COMPLETIONS_TOOLS = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": TOOL_PARAMETERS,
    },
}]

# Responses API shape (flat: name/parameters beside "type"); OpenAI.
RESPONSES_TOOLS = [{
    "type": "function",
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "parameters": TOOL_PARAMETERS,
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

    @staticmethod
    def _clip(text: str, stream: str) -> tuple[str, bool]:
        """Fit `text` to OUTPUT_MAX_CHARS by cutting the MIDDLE. Head-only
        truncation dropped exactly the part a log or a failed build puts
        last, so the model reasoned from the preamble and never saw the
        error. The marker states how much is missing so the model can ask
        for a narrower read instead of assuming it saw everything."""
        if len(text) <= OUTPUT_MAX_CHARS:
            return text, False
        head = OUTPUT_MAX_CHARS // 2
        tail = OUTPUT_MAX_CHARS - head
        omitted = len(text) - head - tail
        return (f"{text[:head]}\n...[{stream} truncated: {omitted} chars "
                f"omitted from the middle]...\n{text[-tail:]}"), True

    def as_tool_payload(self) -> str:
        out, cut_out = self._clip(self.stdout, "stdout")
        err, cut_err = self._clip(self.stderr, "stderr")
        if cut_out or cut_err:
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


def parse_command_args(args: Any) -> tuple[str, str, str | None]:
    """Validate a run_command tool call's arguments -> (command, explanation, error).

    The model controls this payload and the tool schema is not enforced on the
    wire: json.loads() of a bare array or scalar succeeds, so `arguments` can
    be any JSON type (only JSONDecodeError is caught at the provider seam), and
    a field can hold a non-string. The tool loop is not inside the turn's
    try/except, so an AttributeError from a blind .get()/.strip() here escapes
    run_repl() and takes the session down with it. Everything unusable is
    returned as an error string instead and handed back to the model as a tool
    error, which keeps the turn alive and lets it retry.

    A non-string explanation is demoted to "" rather than rejected: the reason
    line is cosmetic, and failing an otherwise valid command over it is worse.
    """
    if not isinstance(args, dict):
        return "", "", (f"malformed tool arguments: expected a JSON object, "
                        f"got {type(args).__name__}")
    raw_cmd = args.get("command")
    if raw_cmd is not None and not isinstance(raw_cmd, str):
        return "", "", (f"malformed 'command': expected a string, "
                        f"got {type(raw_cmd).__name__}")
    raw_why = args.get("explanation")
    why = raw_why.strip() if isinstance(raw_why, str) else ""
    cmd = raw_cmd.strip() if isinstance(raw_cmd, str) else ""
    if not cmd:
        return "", why, "empty command"
    return cmd, why, None


def prompt_approval(cmd: str, explanation: str, auto: bool) -> tuple[str, str | None]:
    """Returns (action, edited_cmd_or_none). action in {run, skip, stop}.

    "run"  execute (optionally the edited command).
    "skip" decline THIS command; the agent turn continues with the next call.
    "stop" abort the whole agent turn and return to the idle prompt. This does
           NOT end the session — session-quit stays explicit (/exit, /quit,
           Ctrl-D at the idle prompt). 'q', Ctrl-C, and Ctrl-D here all stop.
    """
    print()
    print(dim("─" * 72))
    # Only the label is colored; the command text stays plain (user
    # preference). Control characters are shown visibly so a command cannot
    # hide or rewrite part of itself in the approval prompt.
    shown_cmd, hidden = visible(cmd)
    print(f"{cmd_label('COMMAND:')}  {shown_cmd}")
    print(f"REASON:   {visible(explanation, newlines=False)[0]}")
    print(dim(f"CWD:      {os.getcwd()}"))
    print(dim("─" * 72))
    if hidden:
        print(warn_bold(
            f"[warning: this command contains {hidden} control or invisible "
            "character(s), shown above as \\xNN / \\uNNNN escapes; it would "
            "run with the real characters]"))
        if auto:
            # Require review when a command contains hidden characters.
            print(warn("[auto-approve on, but not applied to this command]"))
            auto = False
    if auto:
        print(warn("[auto-approve on] running."))
        return "run", None
    while True:
        try:
            ans = input_no_history(
                warn_bold("Run? [y]es / [n]o / [e]dit / [q]uit-workflow: ")
            ).strip().lower()
        except EOFError:
            # Ctrl-D here cancels the turn, just like q.
            return "stop", None
        except KeyboardInterrupt:
            # Finish the input line so the caller can print the cancellation notice.
            print()
            return "stop", None
        if ans in ("y", "yes", ""):
            return "run", None
        if ans in ("n", "no"):
            return "skip", None
        if ans in ("q", "quit", "stop", "abort"):
            return "stop", None
        if ans in ("e", "edit"):
            # Pre-fill the prompt with the proposed command for editing.
            try:
                new = _input_prefilled("edit> ", cmd).strip()
            except EOFError:
                return "stop", None
            except KeyboardInterrupt:
                # Cancel the edit, fall back to the y/n/e/q prompt.
                print(fail("\n[edit cancelled]"))
                continue
            if new:
                return "run", new


# Give the group time to exit after SIGTERM before using SIGKILL.
# Limit the final wait for the shell so cancellation cannot hang the REPL.
_GROUP_TERM_GRACE = 5.0
_GROUP_KILL_WAIT = 5.0


def _terminate_group(proc: subprocess.Popen) -> None:
    """
    Bounded SIGTERM-then-SIGKILL of the command's entire process group.
    subprocess's own timeout only kills the immediate child (the shell), which
    leaves grandchildren (e.g. apt-get under /bin/sh) orphaned.

    Requires start_new_session=True, which makes the shell the leader of a new
    group: pgid == proc.pid. Using the pid directly instead of
    os.getpgid(proc.pid) keeps cleanup working once the shell has been reaped,
    when that lookup raises ProcessLookupError and would abandon the group.

    Escalation is driven by probing the group with signal 0, never by the
    shell's exit status. The shell almost always dies on the first SIGTERM, so
    waiting on it says nothing about whether the group is clear — a descendant
    that ignores SIGTERM outlives it, and returning on the shell's exit (as
    this once did) meant SIGKILL was never sent. The shell is reaped first so
    its zombie does not keep the group looking alive.

    POSIX only; without killpg this degrades to killing the direct child.
    """
    killpg = getattr(os, "killpg", None)
    if killpg is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return
    pgid = proc.pid                     # group leader, per start_new_session
    try:
        killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + _GROUP_TERM_GRACE
    try:
        # Reap the shell so it stops counting as a live member of the group.
        proc.wait(timeout=_GROUP_TERM_GRACE)
    except subprocess.TimeoutExpired:
        pass
    while True:
        try:
            killpg(pgid, 0)             # anyone left?
        except ProcessLookupError:
            return                      # group is clear; SIGTERM was enough
        except PermissionError:
            break                       # alive but not ours to probe
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    try:
        killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=_GROUP_KILL_WAIT)
    except subprocess.TimeoutExpired:
        pass


def _decode_output(data: bytes) -> str:
    """Bytes -> text the way text-mode pipes did (locale encoding, universal
    newlines), except that undecodable bytes are replaced: a strict decode
    turned any binary output into an "exec error" with nothing shown."""
    text = data.decode(errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def execute(cmd: str) -> CmdResult:
    """Run one approved command in its own session and return its result.

    stdin is /dev/null. Inherited, it was the REPL's terminal: the new
    session has no controlling tty, so nothing stops the child reading it,
    and a command that asked a question (`apt-get` without -y, `rm -i`)
    waited unseen behind the spinner until the timeout while consuming the
    user's keystrokes. With /dev/null such a command gets EOF and fails at
    once, which the model can see and correct.

    Output is read by one thread per stream into buffers capped at
    OUTPUT_CAPTURE_MAX_BYTES. The threads keep draining past the cap so the
    child never blocks on a full pipe; the main thread sees the overflow and
    stops the group. As with communicate(), the command is done only when
    both pipes reach EOF and the shell has exited, so a backgrounded
    descendant holding a pipe still runs into COMMAND_TIMEOUT.
    """
    try:
        proc = subprocess.Popen(
            cmd, shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as e:                  # noqa: BLE001
        return CmdResult(cmd, 1, "", f"exec error: {e}")

    cap = OUTPUT_CAPTURE_MAX_BYTES
    overflow = threading.Event()

    def pump(pipe: Any, buf: bytearray) -> None:
        try:
            while chunk := pipe.read1(65536):
                room = cap - len(buf)
                if room > 0:
                    buf.extend(chunk[:room])
                if len(chunk) > room:
                    overflow.set()
        except (OSError, ValueError):
            pass

    out_buf, err_buf = bytearray(), bytearray()
    pumps = [
        threading.Thread(target=pump, args=(proc.stdout, out_buf),
                         name="exec-stdout", daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, err_buf),
                         name="exec-stderr", daemon=True),
    ]
    for t in pumps:
        t.start()

    def collect() -> tuple[str, str]:
        # Stop waiting if a process outside the group still holds a pipe open.
        # Keep whatever output the reader collected so far.
        for t in pumps:
            t.join(timeout=5)
        if not any(t.is_alive() for t in pumps):
            for pipe in (proc.stdout, proc.stderr):
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass
        return _decode_output(bytes(out_buf)), _decode_output(bytes(err_buf))

    deadline = time.monotonic() + COMMAND_TIMEOUT
    stopped = ""                            # "" | "timeout" | "overflow"
    try:
        while any(t.is_alive() for t in pumps):
            if overflow.is_set():
                stopped = "overflow"
                break
            if time.monotonic() >= deadline:
                stopped = "timeout"
                break
            pumps[0].join(0.05)
            pumps[1].join(0.05)
        if not stopped:
            if overflow.is_set():           # crossed the cap on its last write
                stopped = "overflow"
            else:
                try:
                    proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    stopped = "timeout"
        if stopped:
            _terminate_group(proc)
        out, err = collect()
    except KeyboardInterrupt:
        # Kill the group before unwinding so Ctrl-C never orphans a command.
        _terminate_group(proc)
        raise
    except Exception as e:                  # noqa: BLE001
        _terminate_group(proc)
        return CmdResult(cmd, 1, "", f"exec error: {e}")
    if stopped == "timeout":
        return CmdResult(cmd, 124, out, f"TIMEOUT after {COMMAND_TIMEOUT}s")
    if stopped == "overflow":
        return CmdResult(cmd, 125, out, (
            f"OUTPUT LIMIT: the command wrote more than {cap} bytes and was "
            "stopped. Narrow it (head, tail, grep, -n, --since) and rerun."))
    return CmdResult(cmd, proc.returncode, out, err)


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
        # Anthropic reports cached tokens separately. OpenAI and DeepSeek
        # already include them in input_tokens, so their cache fields stay zero.
        return self.input_tokens + self.cache_read + self.cache_write


@dataclass
class ChatTurn:
    text: str
    tool_calls: list[ToolCall]
    raw_message: Any           # provider-native dict, appended to history
    usage: Usage
    thinking_text: str = ""    # thinking scratchpad (Anthropic/DeepSeek) or OpenAI reasoning summary
    # An empty stop means the model finished normally. Refusals and output
    # limits can leave an unfinished command that still parses. Keep tool_calls
    # so the REPL can distinguish it from cut-off text and block approval.
    # stop_detail explains the stop to the user.
    stop: str = ""
    stop_detail: str = ""


STOP_REFUSAL = "refusal"
STOP_TRUNCATED = "truncated"


def stop_notice(turn: "ChatTurn") -> str:
    """One-line user-facing account of a refused or truncated turn. Empty for
    a normal stop. Callers add the disposition (what was discarded, whether
    anything ran) since that depends on where the turn was consumed."""
    if turn.stop == STOP_REFUSAL:
        return (f"[model declined: {turn.stop_detail} — try rephrasing, "
                "/model <other>, or /consult for a second opinion]")
    if turn.stop == STOP_TRUNCATED:
        return f"[reply truncated — {turn.stop_detail}]"
    return ""


class Provider:
    name: str = ""
    model: str = ""
    # Anthropic caches the growing live conversation. /consult sends the
    # transcript once, so it skips the cache write.
    cache_conversation: bool = True

    def initial_messages(self, system: str) -> list[dict]:
        raise NotImplementedError

    def chat(
        self, messages: list[dict], system: str,
        thinking: bool = False, effort: str = "high",
    ) -> ChatTurn:
        raise NotImplementedError

    def append_assistant(self, messages: list[dict], turn: ChatTurn) -> None:
        messages.append(turn.raw_message)

    def append_tool_results(
        self, messages: list[dict], results: list[tuple[str, str]]
    ) -> None:
        raise NotImplementedError

    def render_assistant(self, text: str, tool_calls: list["ToolCall"]) -> dict:
        # Convert a shared transcript turn to the receiving provider's format.
        raise NotImplementedError

    def append_rendered(
        self, messages: list[dict], text: str, tool_calls: list["ToolCall"]
    ) -> None:
        # Responses overrides this because one turn can span several input items.
        messages.append(self.render_assistant(text, tool_calls))

    # Each provider reports the thinking controls its model supports.

    def _honors_thinking(self) -> bool:
        """The model thinks when the session flag is on."""
        return False

    def _always_thinks(self) -> bool:
        """The model thinks even with the flag off (it cannot be disabled)."""
        return False

    def _thinking_off_status(self) -> str:
        return "off"

    def thinking_active(self, flag: bool) -> bool:
        """True when thinking will actually run. Displays report effective
        state through this rather than the raw flag, so they never imply
        thinking is running when it is not (or, for always-on models, not
        running when it is)."""
        return self._always_thinks() or (flag and self._honors_thinking())

    def thinking_status(self, flag: bool) -> str:
        """Display text: off / on / on-but-inactive. Models that cannot
        disable thinking read as on with the flag off, so /thinking off never
        claims a state the model is not in."""
        if not flag:
            return self._thinking_off_status()
        if self._honors_thinking():
            return "on"
        return f"on (inactive: {self.name}/{self.model})"

    def effort_applies(self) -> bool:
        """True when the model honors the session effort setting."""
        return False

    def effective_effort(self, effort: str) -> tuple[str, str]:
        """(grade actually sent, note) for a requested effort; note is ""
        unless the provider changed the value. Display-only: the stored
        session effort keeps the user's request, so switching to a
        finer-grained provider still honors it."""
        return effort, ""

    def thinking_mode(self) -> str | None:
        """The wire control /info reports, or None when there is none."""
        return None


class OpenAIProvider(Provider):
    """OpenAI via the Responses API (/v1/responses), for every OpenAI model.

    Why Responses and not Chat Completions: gpt-6-astra and gpt-6.1-sol
    cannot call function tools on Chat Completions at all, and every other
    current model there only with reasoning_effort "none" (probes recorded
    in docs/models.md). One endpoint for the whole provider
    keeps one wire format per provider name, which /model relies on (it
    switches model in place and keeps `messages`) and normalize() keys on.

    Wire format: `messages` holds Responses input items. User turns are
    {"role": "user", "content": str}; an assistant turn is the response's
    output items verbatim (reasoning, message, function_call), so one turn
    adds SEVERAL items (append_assistant extends); a tool result is a
    function_call_output keyed on call_id. The system prompt travels as
    `instructions` on every call, so initial_messages() is empty.

    Stateless: store=False, the conversation is resent each call (as on
    every other provider). Reasoning items come back with encrypted_content
    by default when store=False, with no `include` needed (probe
    2026-11-16).

    REASONING REPLAY (soft, unlike DeepSeek's): reasoning items are replayed
    verbatim beside their function_calls. Omitting them does NOT 400; the
    model silently re-reasons (Astra: 0 reasoning tokens on the next call
    with them, 55 without; probe 2026-11-16). So dropping them costs tokens
    and consistency, not correctness. Items stripped of `id` or `status`
    are accepted, which is what lets render() rebuild function_calls for a
    transcript another provider wrote. A reasoning model's items replayed
    to a different OpenAI model (gpt-6.1-sol -> gpt-6-luna) are accepted
    too, so /model mid-session needs no history rewrite. From GPT-6.2 the
    server silently drops reasoning items after an edit BEFORE them or on a
    downgrade; whole-item deletion and appending do not trigger it, so the
    REPL's trailing-only rollbacks keep reasoning (probes 2026-11-17).
    """

    def __init__(self, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("openai SDK missing — uv should install via PEP 723 deps")
        self.client = OpenAI(max_retries=API_MAX_RETRIES)
        self.model = model
        self.name = "openai"

    def initial_messages(self, system: str) -> list[dict]:
        # System prompt goes in `instructions` on each call, not in input.
        return []

    def _honors_thinking(self) -> bool:
        return self.model in _OPENAI_REASONING_EFFORTS

    def _always_thinks(self) -> bool:
        return _openai_always_reasons(self.model)

    def _thinking_off_status(self) -> str:
        if self._always_thinks():
            return f"on (always on for {self.model}: lowest effort is low)"
        return "off"

    def effort_applies(self) -> bool:
        return self._honors_thinking()

    def effective_effort(self, effort: str) -> tuple[str, str]:
        sent = _openai_effort(self.model, True, effort)
        if sent is not None and sent != effort:
            return sent, f"capped at {self.model}'s highest grade"
        return effort, ""

    def thinking_mode(self) -> str | None:
        return "reasoning.effort" if self._honors_thinking() else None

    def chat(
        self, messages: list[dict], system: str,
        thinking: bool = False, effort: str = "high",
    ) -> ChatTurn:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            instructions=system,
            input=messages,
            tools=RESPONSES_TOOLS,
            tool_choice="auto",
            store=False,
        )
        eff = _openai_effort(self.model, thinking, effort)
        if eff is not None:
            reasoning: dict[str, str] = {"effort": eff}
            # Ask for a visible summary only when /thinking is on.
            if thinking and eff != "none":
                reasoning["summary"] = "auto"
            kwargs["reasoning"] = reasoning
        resp = self.client.responses.create(**kwargs)
        if resp.status not in ("completed", "incomplete"):
            # Treat failed or cancelled responses like API errors in the REPL.
            detail = getattr(getattr(resp, "error", None), "message", "") or ""
            raise RuntimeError(f"OpenAI response {resp.status}: {detail}".rstrip(": "))
        items = [it.model_dump(exclude_none=True) for it in resp.output]
        text_parts: list[str] = []
        summary_parts: list[str] = []
        refusal = ""
        calls: list[ToolCall] = []
        for it in items:
            kind = it.get("type")
            if kind == "function_call":
                try:
                    args = json.loads(it.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                # Pair tool calls and results by call_id.
                calls.append(ToolCall(id=it.get("call_id", ""),
                                      name=it.get("name", ""), arguments=args))
            elif kind == "message":
                for part in it.get("content") or []:
                    if part.get("type") == "output_text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "refusal":
                        refusal = part.get("refusal", "") or "refused"
            elif kind == "reasoning":
                summary_parts.extend(s.get("text", "")
                                     for s in it.get("summary") or [])
        stop, stop_detail = self._classify_status(resp, refusal)
        usage = resp.usage
        return ChatTurn(
            text="".join(text_parts),
            tool_calls=calls,
            raw_message=items,
            # Cached tokens are already included in input_tokens.
            usage=Usage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            ),
            thinking_text="\n\n".join(p for p in summary_parts if p),
            stop=stop,
            stop_detail=stop_detail,
        )

    def _classify_status(self, resp: Any, refusal: str) -> tuple[str, str]:
        """Map a Responses result to (ChatTurn.stop, detail).

        A refusal is a `refusal` content part inside the output message
        (ResponseOutputRefusal, openai SDK 3.51), on an otherwise completed
        response. Truncation is status "incomplete" with
        incomplete_details.reason one of max_output_tokens, max_messages,
        content_filter, steered (SDK 3.51 IncompleteDetails; steered is
        WebSocket-only). sys_agent sends no max_output_tokens, so
        max_output_tokens means the model's own ceiling. Anything incomplete
        that is not a filter is treated as truncated: a function_call cut
        short still parses, so the REPL must discard it.
        """
        if refusal:
            return STOP_REFUSAL, f"openai refusal: {refusal[:200]}"
        if resp.status != "incomplete":
            return "", ""
        reason = getattr(getattr(resp, "incomplete_details", None),
                         "reason", None)
        if reason == "content_filter":
            return STOP_REFUSAL, "openai content_filter"
        if reason == "max_output_tokens":
            return STOP_TRUNCATED, (
                f"hit openai's output-token limit for {self.model}; "
                "ask for a shorter answer or fewer steps and resend")
        return STOP_TRUNCATED, (
            f"openai response incomplete ({reason or 'no reason given'}); "
            "resend")

    def append_assistant(self, messages: list[dict], turn: ChatTurn) -> None:
        # raw_message is the list of output items (see class docstring).
        messages.extend(turn.raw_message)

    def append_tool_results(
        self, messages: list[dict], results: list[tuple[str, str]]
    ) -> None:
        for call_id, content in results:
            messages.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": content,
            })

    def append_rendered(
        self, messages: list[dict], text: str, tool_calls: list["ToolCall"]
    ) -> None:
        # A turn another provider authored: no reasoning items exist for it,
        # and items without id/status are accepted (class docstring).
        if text:
            messages.append({"type": "message", "role": "assistant",
                             "content": [{"type": "output_text", "text": text}]})
        for tc in tool_calls:
            messages.append({"type": "function_call", "call_id": tc.id,
                             "name": tc.name,
                             "arguments": json.dumps(tc.arguments)})


class ChatCompletionsProvider(Provider):
    """Wire format of an OpenAI-compatible /chat/completions endpoint: system
    prompt as the first message, assistant tool_calls, one `role: tool`
    message per result. DeepSeek is the only user since OpenAI moved to the
    Responses API; subclasses supply __init__ and chat()."""

    def initial_messages(self, system: str) -> list[dict]:
        return [{"role": "system", "content": system}]

    def _classify_finish(self, choice: Any) -> tuple[str, str]:
        """Map Chat Completions finish_reason to (ChatTurn.stop, detail).

        Documented values (openai SDK 3.19 Choice.finish_reason): stop,
        tool_calls, function_call (normal); `length` = max_tokens or the
        model's output limit reached; `content_filter` = content omitted by
        the provider's filter. sys_agent sends no max_tokens on this path, so
        `length` means the model's own output ceiling — the remedy is a
        smaller ask, not a config change.
        """
        reason = getattr(choice, "finish_reason", None)
        if reason == "length":
            return STOP_TRUNCATED, (
                f"hit {self.name}'s output-token limit for {self.model}; "
                "ask for a shorter answer or fewer steps and resend")
        if reason == "content_filter":
            return STOP_REFUSAL, f"{self.name} content_filter"
        return "", ""

    def append_tool_results(
        self, messages: list[dict], results: list[tuple[str, str]]
    ) -> None:
        # One `role: tool` message per tool_call_id.
        for tc_id, content in results:
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": content,
            })

    def render_assistant(self, text: str, tool_calls: list["ToolCall"]) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = [{
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name,
                             "arguments": json.dumps(tc.arguments)},
            } for tc in tool_calls]
        return msg


class DeepSeekProvider(ChatCompletionsProvider):
    """DeepSeek V4.x (deepseek-flash; v4-* names server-routed) via the
    OpenAI-compatible ChatCompletions API.

    Inherits the Chat Completions wire format (initial_messages,
    append_tool_results, render_assistant, _classify_finish) from
    ChatCompletionsProvider and supplies:

      1. __init__ pointing the openai SDK at DeepSeek's base_url with the
         DEEPSEEK_API_KEY (the SDK would otherwise default to OPENAI_API_KEY).
      2. chat(), which honors extended thinking. DeepSeek
         exposes it via reasoning_effort + thinking={"type":"enabled"} and
         returns the scratchpad in a `reasoning_content` field beside `content`.
         Thinking is ON by default server-side, so the off path must send
         thinking={"type":"disabled"} explicitly — omitting the field thinks.

    THINKING-MODE REPLAY CONTRACT (the subtle part): once a thinking-enabled
    turn makes a tool call, DeepSeek REQUIRES `reasoning_content` to be present
    on every assistant message replayed in subsequent requests, or the next
    call fails with 400 "reasoning_content ... must be passed back". Because
    this agent tool-calls on essentially every turn, that path is the norm, not
    an edge case — it's the bug that broke Claude Code / Cursor / Roo on the V4
    launch. So reasoning_content is preserved verbatim on the raw assistant
    message (defaulting to "" — which DeepSeek ignores — when a turn produced
    none, satisfying the all-or-nothing presence rule). This mirrors how the
    Anthropic path must echo back thinking blocks with their signatures.
    """

    # Round unsupported levels up: medium -> high (as DeepSeek does), and
    # xhigh -> max, where DeepSeek's own table says high, so /effort xhigh
    # stays distinct from high.
    _EFFORT_MAP: dict[str, str] = {
        "low": "low", "medium": "high", "high": "high",
        "xhigh": "max", "max": "max",
    }

    @staticmethod
    def _map_effort(effort: str) -> str:
        return DeepSeekProvider._EFFORT_MAP.get(effort, "high")

    def __init__(self, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("openai SDK missing — uv should install via PEP 723 deps")
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        self.client = OpenAI(
            api_key=key, base_url=DEEPSEEK_BASE_URL, max_retries=API_MAX_RETRIES
        )
        self.model = model
        self.name = "deepseek"

    def _honors_thinking(self) -> bool:
        return True

    def effort_applies(self) -> bool:
        return True

    def effective_effort(self, effort: str) -> tuple[str, str]:
        # Three grades (low|high|max); medium and xhigh round up (_EFFORT_MAP).
        mapped = self._map_effort(effort)
        return mapped, ("" if mapped == effort else "rounded up to DeepSeek grade")

    def thinking_mode(self) -> str | None:
        return "reasoning_effort"

    def chat(
        self, messages: list[dict], system: str,
        thinking: bool = False, effort: str = "high",
    ) -> ChatTurn:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            tools=CHAT_COMPLETIONS_TOOLS,
            tool_choice="auto",
        )
        # Use extra_body so older SDKs can send these settings. DeepSeek
        # thinks by default, so /thinking off must disable it explicitly.
        if thinking:
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled"},
                "reasoning_effort": self._map_effort(effort),
            }
        else:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        reasoning = getattr(msg, "reasoning_content", None) or ""
        raw = msg.model_dump(exclude_none=True)
        # Keep reasoning_content on every assistant message for later tool calls.
        raw["reasoning_content"] = reasoning
        stop, stop_detail = self._classify_finish(choice)
        return ChatTurn(
            text=msg.content or "",
            tool_calls=calls,
            raw_message=raw,
            usage=Usage(
                input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            ),
            thinking_text=reasoning,
            stop=stop,
            stop_detail=stop_detail,
        )

    def render_assistant(self, text: str, tool_calls: list["ToolCall"]) -> dict:
        # DeepSeek requires reasoning_content when replaying tool use.
        # Use an empty string for turns from another provider.
        msg = super().render_assistant(text, tool_calls)
        msg["reasoning_content"] = ""
        return msg


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

    def _honors_thinking(self) -> bool:
        return True

    def _always_thinks(self) -> bool:
        return _thinking_always_on(self.model)

    def _thinking_off_status(self) -> str:
        if self._always_thinks():
            return f"on (always on for {self.model})"
        if _thinking_unverified(self.model):
            return f"off (unverified for {self.model}: it may think by default)"
        off = _thinking_off_payload(self.model)
        if off and off.get("type") == "between_tools":
            return "off (between_tools: notes between tool calls still shown)"
        return "off"

    def effort_applies(self) -> bool:
        # Adaptive models honor all five grades verbatim; legacy (Haiku 4.5)
        # thinking ignores effort.
        return _thinking_mode(self.model) == "adaptive"

    def thinking_mode(self) -> str | None:
        return _thinking_mode(self.model)

    def chat(
        self, messages: list[dict], system: str,
        thinking: bool = False, effort: str = "high",
    ) -> ChatTurn:
        # Cache the system prompt separately so it can still be reused after
        # a rollback or a change to thinking settings.
        system_blocks = [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
        kwargs: dict[str, Any] = dict(
            model=self.model,
            system=system_blocks,
            messages=messages,
            tools=ANTHROPIC_TOOLS,
            max_tokens=ANTHROPIC_MAX_TOKENS,
        )
        if thinking:
            # Thinking requires automatic tool choice and more output tokens.
            kwargs["max_tokens"] = ANTHROPIC_THINKING_MAX_TOKENS
            if _thinking_mode(self.model) == "adaptive":
                # Adaptive thinking works across tool calls without a beta
                # header. Use extra_body for older SDKs.
                kwargs["extra_body"] = {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": effort},
                }
            else:
                # The legacy beta lets reasoning span tools and exceed max_tokens.
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": ANTHROPIC_THINKING_BUDGET,
                }
                kwargs["extra_headers"] = {
                    "anthropic-beta": ANTHROPIC_INTERLEAVED_BETA
                }
        elif _thinking_headroom(self.model):
            # Use the server's defaults. Always-thinking and unverified
            # models may still need the larger token limit.
            kwargs["max_tokens"] = ANTHROPIC_THINKING_MAX_TOKENS
        elif (off := _thinking_off_payload(self.model)) is not None:
            # Some models need an explicit setting for /thinking off.
            # Let the server choose an effort compatible with that setting.
            kwargs["extra_body"] = {"thinking": dict(off)}
        if self.cache_conversation:
            # Advance the cache marker so Anthropic reuses earlier messages
            # and caches only the new ones; without it every tool-loop call
            # re-billed the whole history at the full input rate. It uses one
            # of the four breakpoints and keeps the system marker's 5-minute
            # lifetime, which the API requires to be no longer than the one
            # before it. Sent via extra_body for older SDKs. After a rollback
            # the API reuses an earlier entry within 20 blocks; changing
            # thinking or effort rebuilds the message cache once.
            kwargs.setdefault("extra_body", {})["cache_control"] = {
                "type": "ephemeral"}
        if thinking or _thinking_headroom(self.model):
            # The SDK requires streaming at this token limit. Reassemble
            # the response into the same Message object used below.
            with self.client.messages.stream(**kwargs) as stream:
                resp = stream.get_final_message()
        else:
            resp = self.client.messages.create(**kwargs)
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        calls: list[ToolCall] = []
        raw_content: list[dict] = []
        for block in resp.content:
            # Preserve the original block order, reasoning signatures, and
            # redacted thinking so the API accepts the next tool result.
            raw_content.append(block.model_dump(exclude_none=True))
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif block.type == "tool_use":
                calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.input or {}),
                ))
        stop, stop_detail = self._classify_stop(resp, kwargs["max_tokens"])
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
            thinking_text="\n".join(thinking_parts),
            stop=stop,
            stop_detail=stop_detail,
        )

    def _classify_stop(self, resp: Any, max_tokens: int) -> tuple[str, str]:
        """Map Message.stop_reason to (ChatTurn.stop, detail).

        Documented values (anthropic SDK 1.8 StopReason): end_turn, tool_use,
        stop_sequence, pause_turn (normal here — no server tools are used);
        `max_tokens` = the requested cap or the model's maximum was reached;
        `model_context_window_exceeded`; `refusal` = a classifier intervened
        (Opus 5 cyber; Opus 5.5 / Fable 5.x also bio, reasoning_extraction,
        frontier_llm, general_harms) or the model declined. Branch on
        stop_reason only: stop_details is populated solely on refusal and
        both its category and explanation may be null even then.

        `max_tokens` here is the cap this request actually sent — 4096 on a
        plain turn, ANTHROPIC_THINKING_MAX_TOKENS on a thinking or always-on
        turn — so the remedy names the knob that would have helped. /effort is
        offered only where it is honored (adaptive-thinking models); on the
        legacy budget path (Haiku 4.5) it is a no-op, and /effort itself says
        so, so naming it there sends the user to a dead knob.
        """
        reason = getattr(resp, "stop_reason", None)
        if reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            explanation = getattr(details, "explanation", None)
            detail = f"category={category}"
            if explanation:
                detail += f" — {explanation}"
            return STOP_REFUSAL, detail
        if reason == "max_tokens":
            if max_tokens >= ANTHROPIC_THINKING_MAX_TOKENS:
                remedy = "raise SYS_THINKING_MAX_TOKENS"
                if _thinking_mode(self.model) == "adaptive":
                    remedy += " or lower /effort"
            else:
                remedy = (f"/thinking on raises it to "
                          f"{ANTHROPIC_THINKING_MAX_TOKENS} "
                          "(SYS_THINKING_MAX_TOKENS)")
            return STOP_TRUNCATED, (
                f"hit the {max_tokens}-token output cap; {remedy}, "
                "then resend")
        if reason == "model_context_window_exceeded":
            return STOP_TRUNCATED, (
                "the context window is full; /reset starts a fresh "
                "conversation")
        return "", ""

    def append_tool_results(
        self, messages: list[dict], results: list[tuple[str, str]]
    ) -> None:
        # Anthropic requires one user message with all results for the turn.
        blocks = [
            {"type": "tool_result", "tool_use_id": tc_id, "content": content}
            for tc_id, content in results
        ]
        messages.append({"role": "user", "content": blocks})

    def render_assistant(self, text: str, tool_calls: list["ToolCall"]) -> dict:
        # Replay only text and tool calls from the shared transcript.
        # Anthropic requires a content block, so use a space for empty turns.
        content: list[dict] = []
        if text:
            content.append({"type": "text", "text": text})
        for tc in tool_calls:
            content.append({"type": "tool_use", "id": tc.id,
                            "name": tc.name, "input": tc.arguments})
        return {"role": "assistant",
                "content": content or [{"type": "text", "text": " "}]}


# -----------------------------------------------------------------------------
# Shared transcript for /consult
# -----------------------------------------------------------------------------
#
# /consult translates the live history through a shared transcript format.
# Each provider structures messages and tool results differently, so build
# the transcript from the active history on demand. Reasoning blocks stay
# with the original provider because they may be signed or encrypted.


@dataclass
class CanonicalEvent:
    kind: str                                   # user | assistant | tool_results
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[tuple[str, str]] = field(default_factory=list)


def normalize(provider_name: str, messages: list[dict]) -> list[CanonicalEvent]:
    """Read the active provider's `messages` into provider-neutral events.

    Consecutive tool results (DeepSeek `role:tool` messages, OpenAI
    function_call_output items) are coalesced into one tool_results event so
    the canonical form matches Anthropic's batching — the neutral shape is
    symmetric regardless of which provider produced it."""
    events: list[CanonicalEvent] = []
    if provider_name == "openai":
        # Join consecutive Responses assistant items into one turn. Skip reasoning.
        def _asst() -> CanonicalEvent:
            if not events or events[-1].kind != "assistant":
                events.append(CanonicalEvent("assistant"))
            return events[-1]
        for m in messages:
            kind = m.get("type")
            if kind is None and m.get("role") == "user":
                events.append(CanonicalEvent("user", text=m.get("content") or ""))
            elif kind == "message" and m.get("role") == "assistant":
                text = "".join(p.get("text", "") for p in m.get("content") or []
                               if p.get("type") == "output_text")
                ev = _asst()
                ev.text = "\n".join(t for t in (ev.text, text) if t)
            elif kind == "function_call":
                try:
                    args = json.loads(m.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                _asst().tool_calls.append(ToolCall(
                    id=m.get("call_id", ""), name=m.get("name", ""),
                    arguments=args))
            elif kind == "reasoning":
                _asst()
            elif kind == "function_call_output":
                pair = (m.get("call_id", ""), m.get("output") or "")
                if events and events[-1].kind == "tool_results":
                    events[-1].results.append(pair)
                else:
                    events.append(CanonicalEvent("tool_results", results=[pair]))
        return events

    if provider_name == "deepseek":
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            if role == "user":
                events.append(CanonicalEvent("user", text=m.get("content") or ""))
            elif role == "assistant":
                calls: list[ToolCall] = []
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    calls.append(ToolCall(id=tc.get("id", ""),
                                          name=fn.get("name", ""), arguments=args))
                events.append(CanonicalEvent("assistant",
                                             text=m.get("content") or "",
                                             tool_calls=calls))
            elif role == "tool":
                pair = (m.get("tool_call_id", ""), m.get("content") or "")
                if events and events[-1].kind == "tool_results":
                    events[-1].results.append(pair)
                else:
                    events.append(CanonicalEvent("tool_results", results=[pair]))
        return events

    # Anthropic: assistant content is a block list; tool_results arrive as a
    # user message whose content is a list of tool_result blocks.
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            if isinstance(content, str):
                events.append(CanonicalEvent("user", text=content))
            elif isinstance(content, list):
                results = [(b.get("tool_use_id", ""), b.get("content") or "")
                           for b in content if b.get("type") == "tool_result"]
                if results:
                    events.append(CanonicalEvent("tool_results", results=results))
                else:
                    text = "".join(b.get("text", "") for b in content
                                   if b.get("type") == "text")
                    events.append(CanonicalEvent("user", text=text))
        elif role == "assistant":
            text_parts: list[str] = []
            calls = []
            for b in (content or []):
                if b.get("type") == "text":
                    text_parts.append(b.get("text", ""))
                elif b.get("type") == "tool_use":
                    calls.append(ToolCall(id=b.get("id", ""), name=b.get("name", ""),
                                          arguments=dict(b.get("input") or {})))
            events.append(CanonicalEvent("assistant",
                                         text="\n".join(text_parts), tool_calls=calls))
    return events


def render(target: Provider, events: list[CanonicalEvent], system: str) -> list[dict]:
    """Replay neutral events into `target`'s wire format. Reuses the provider's
    own initial_messages / render_assistant / append_tool_results so format
    quirks (system-prompt placement, tool_result batching, reasoning_content
    presence) stay owned by the provider, not duplicated here."""
    out = target.initial_messages(system)
    for ev in events:
        if ev.kind == "user":
            out.append({"role": "user", "content": ev.text})
        elif ev.kind == "assistant":
            target.append_rendered(out, ev.text, ev.tool_calls)
        elif ev.kind == "tool_results":
            target.append_tool_results(out, ev.results)
    return out





def available_provider_names() -> list[str]:
    names: list[str] = []
    if os.environ.get("OPENAI_API_KEY"):
        names.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        names.append("anthropic")
    if os.environ.get("DEEPSEEK_API_KEY"):
        names.append("deepseek")
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
    if normalized in ("deepseek", "d", "ds"):
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY is not set")
        return DeepSeekProvider(model or DEFAULT_DEEPSEEK_MODEL)
    raise ValueError("provider must be openai, anthropic, or deepseek")


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
    available = available_provider_names()
    if not available:
        sys.exit("error: no API key set (need one of OPENAI_API_KEY, "
                 "ANTHROPIC_API_KEY, DEEPSEEK_API_KEY)")

    # Canonicalize the SYS_PROVIDER aliases the rest of the code accepts.
    aliases = {
        "o": "openai", "a": "anthropic", "claude": "anthropic",
        "d": "deepseek", "ds": "deepseek",
    }

    # Honor SYS_PROVIDER override (skips interactive prompt).
    pref = os.environ.get("SYS_PROVIDER", "").strip().lower()
    if pref:
        canon = aliases.get(pref, pref)
        if canon not in DEFAULT_MODELS:
            sys.exit(f"error: SYS_PROVIDER={pref!r} is not a known provider "
                     "(openai|anthropic|deepseek)")
        if canon not in available:
            sys.exit(f"error: SYS_PROVIDER={canon} but its API key is not set")
        return make_provider(canon)

    # Only one key available — no prompt needed.
    if len(available) == 1:
        only = available[0]
        print(dim(f"[only {only} key available — using {only}]"))
        return make_provider(only)

    # Offer only providers with configured keys.
    labels = {"openai": "OpenAI", "anthropic": "Anthropic", "deepseek": "DeepSeek"}
    print(banner("Select LLM provider:"))
    for idx, name in enumerate(available, 1):
        print(f"  [{idx}] {labels[name]:<10} (model: {DEFAULT_MODELS[name]})")
    while True:
        try:
            ans = input_no_history(ask(f"Choice [1-{len(available)}]: ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nno selection; exiting")
        if ans.isdigit() and 1 <= int(ans) <= len(available):
            return make_provider(available[int(ans) - 1])
        canon = aliases.get(ans, ans)
        if canon in available:
            return make_provider(canon)


# -----------------------------------------------------------------------------
# REPL
# -----------------------------------------------------------------------------

def _redact_api_error(text: str) -> str:
    """Transport errors can embed Authorization header bytes, including keys."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        value = os.environ.get(name, "")
        if value:
            for form in (value, repr(value)[1:-1]):
                text = text.replace(form, "<redacted>")
    text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer <redacted>", text)
    return re.sub(r"\bsk-[A-Za-z0-9_-]+", "<redacted>", text)


def explain_api_error(e: Exception) -> str:
    """
    Turn a raw SDK exception into a short, actionable message. The SDK has
    already exhausted API_MAX_RETRIES by the time this is called, so these
    are persistent failures, not transient blips.
    """
    chain: list[BaseException] = []
    seen: set[int] = set()
    cause: BaseException | None = e
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        chain.append(cause)
        cause = (cause.__cause__ if cause.__cause__ is not None else
                 None if cause.__suppress_context__ else cause.__context__)
    if any("illegal header value" in str(cause).lower() for cause in chain):
        return ("Invalid API request header — check for newlines or other "
                "invalid characters in API keys and custom headers. "
                "Shell-exported variables override the env file; correct the "
                "export or unset the malformed variable, then restart sys_agent.")
    s = _redact_api_error(str(e))
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
    if (any(cls.__name__ in ("APIConnectionError", "APITimeoutError")
            for cls in type(e).__mro__)
            or any(word in s.lower() for word in
                   ("connection", "timeout", "timed out"))):
        # Follow the exception chain to find the original network error.
        details: list[str] = []
        for cause in chain:
            detail = _redact_api_error(
                f"{type(cause).__name__}: {cause}".rstrip(": "))
            if not details or detail != details[-1]:
                details.append(detail)
        return ("Network error reaching the API — check DNS, TLS certificates, "
                "proxy settings, and connectivity.\n"
                "  Cause: " + " -> ".join(details))
    return s          # fall back to the raw message for anything unrecognized


def unlisted_model_warning(provider_name: str, model: str) -> str:
    """Warning for a prefix-matched model absent from PROVIDER_MODELS.

    The context-% hedge applies only when CONTEXT_WINDOWS lacks the model;
    deliberately-unlisted models (claude-fable-5, gpt-5.6-sol, gpt-6.2-astra)
    have windows recorded and display context-% normally.
    """
    tail = ("" if model in CONTEXT_WINDOWS
            else "; context-% may be unavailable")
    if provider_name == "anthropic" and _thinking_unverified(model):
        tail += ("; its thinking default is unverified, so /thinking off "
                 "may not stop it")
    return (f"[warning: {model!r} is not a known {provider_name} model — "
            f"switching anyway{tail}]")


def effort_display(provider: Provider, effort: str, thinking_on: bool) -> tuple[str, str]:
    """(shown_grade, note) for any effort readout — the single source of truth
    shared by the /effort set-confirmation and the bare readout, so the two
    cannot drift. `note` is the parenthetical text (no parens), "" when there
    is nothing to flag:
      - provider ignores effort entirely  -> "no effect on this provider/model"
      - DeepSeek rounded the grade         -> "rounded up to DeepSeek grade"
      - effort honored but thinking is off -> append "needs /thinking on"
    """
    if not provider.effort_applies():
        return effort, "no effect on this provider/model"
    shown, clamp = provider.effective_effort(effort)
    notes = [clamp] if clamp else []
    if not thinking_on:
        notes.append("needs /thinking on")
    return shown, "; ".join(notes)


def _effort_usage_levels(provider: Provider) -> tuple[str, ...]:
    """Distinct effort grades worth offering for the active provider, in
    order. DeepSeek collapses the five inputs to three effective grades, so a
    flat low|medium|high|xhigh|max hint overstates the choices; this returns
    ("low", "high", "max") there and all five elsewhere. All five remain
    *accepted* inputs everywhere (DeepSeek rounds); this only shapes the hint."""
    seen: list[str] = []
    for lvl in _VALID_EFFORTS:
        grade, _ = provider.effective_effort(lvl)
        if grade not in seen:
            seen.append(grade)
    return tuple(seen)


def print_help(
    provider: Provider,
    auto_approve: bool,
    show_tokens: bool,
    thinking_enabled: bool,
    effort: str,
) -> None:
    """Print the meta-command reference plus current toggle states."""
    print(banner("Meta-commands:"))
    width = max(len(cmd) for cmd, _ in META_COMMANDS)
    for cmd, desc in META_COMMANDS:
        print(f"  {cmd.ljust(width)}   {dim(desc)}")
    print()
    print(f"  {PASTE_DELIM} ... {PASTE_DELIM}   " + dim(
        "Multi-line message: open with \"\"\", paste, close with \"\"\" "
        "(Ctrl-C cancels)"))
    print()
    print(dim(
        f"  state: auto-approve={auto_approve}  "
        f"show-tokens={show_tokens}  "
        f"thinking={provider.thinking_status(thinking_enabled)}  "
        f"effort={effort}  "
        f"color={_color_enabled}  "
        f"audit={'on' if _audit_path else 'off'}"
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
        - `sudo` (when present) says whether an approved command can use
          sudo. With no tty, sudo can never prompt for a password.
          `passwordless`: sudo works. `password_required`: any sudo command
          fails at once ("a terminal is required to read the password"), so
          do NOT propose one expecting it to run. Try the unprivileged form
          first (many status and log reads need no root). If root is
          genuinely required, give the exact command in plain text for the
          user to run in their own terminal and ask them to paste back the
          output. Never ask for the user's password or work around the prompt
          (`sudo -S`, an askpass helper, a password piped or echoed in).
          `running_as_root`: commands already run as root; omit sudo.
          `not_installed`: there is no sudo here. Key absent: the startup
          probe was inconclusive; a single attempt will show which case holds.
        - After receiving command output, summarize the findings and decide
          the next step. If the next step is another command, issue it as a
          run_command call in the same turn rather than describing it or
          asking whether to proceed. This applies to read-only investigation
          and to steps of a change the user has already requested — it does
          NOT authorize initiating a state-modifying action the user has not
          asked for.
        - `running_services`, `failed_services`, and `top_processes` (when
          present) are a snapshot taken at startup, not live state: a process
          hot at launch may be idle now, and the service lists are from startup.
          Use them to get real unit/process names right the first time (avoid
          guessing) and for orientation. They are NOT a substitute for
          measurement: any question about current state (memory/disk/CPU
          pressure, load, free space, what is running now) must be answered from
          a live command this turn (`free -h`, `df -h`, `uptime`, `ps`/`top`),
          not from the snapshot. `/facts refresh` re-probes if they look stale.
        - `failed_services` (when present) are SYSTEM units in a failed state at
          probe time — high-signal for triage. Confirm a unit is still failed
          and read the cause with `systemctl status <unit>` and
          `journalctl -u <unit> -n 100 --no-pager` before acting, since a unit
          may have been restarted since startup. An empty/absent list is not a
          guarantee of health (user-scope failures are not probed; check
          `systemctl --user --failed` when a user unit is suspect).
        - `timezone`, `boot_time`, and `uptime_hours` are stable orientation
          facts captured at startup: use them to interpret log timestamps and
          gauge reboot recency without a probe. For current wall-clock or live
          uptime, still run `date` / `uptime`.
        - `hardware.swap_gb` is configured swap; `0` means none. On a low-RAM
          host, zero swap means memory pressure ends in OOM-kills rather than
          swapping — weigh that when diagnosing killed or restarting processes.
        - `memory_policy` (Linux) is effective kernel state read at startup,
          not configuration. `cgroup_memory: false` means the memory
          controller is off: systemd `MemoryHigh=`/`MemoryMax=` and container
          memory limits are NOT enforced, even though `systemctl show` still
          prints the configured values — never cite them as protection then.
          (Raspberry Pi firmware passes `cgroup_disable=memory`; re-enabling
          takes `cgroup_enable=memory` in `/boot/firmware/cmdline.txt` plus a
          reboot — a state change, only when asked.) When it is true, check a
          capped unit with `systemctl [--user] show -p
          MemoryCurrent,MemoryHigh,MemoryMax <unit>` and its counters in
          `/sys/fs/cgroup$(systemctl [--user] show -p ControlGroup --value
          <unit>)/memory.events` (`high` = reclaim throttling, `max`/`oom_kill`
          = hard-limit hits). `psi: true` makes `/proc/pressure/memory` (and a
          cgroup's `memory.pressure`) the first measure of memory pressure:
          stall time, which free memory alone does not show; `psi: false`
          means those files are unavailable, so do not propose them.
          `swappiness` is the live value and wins over whatever a file under
          `/etc/sysctl.d` says (on Debian, `/etc/sysctl.conf` loads as
          `99-sysctl.conf`, after most of them). `swap_kinds` and `zswap` say
          what backs swap.
        - `hardware.cloud` (when present, e.g. aws|gcp|azure|oci) classifies the
          cloud platform from DMI — a platform fact, not identity. Use it to
          pick strategy: cloud VMs have network-attached storage (a full root
          disk is a volume resize via the provider, not local `parted` on
          physical media) and provider-level firewalling (security groups /
          NACLs) layered on top of host `iptables`/`ufw`, so a connectivity or
          capacity question may have its real answer at the provider, not on the
          host. Instance metadata (region, AZ, instance-type, instance-id, IAM
          role) is deliberately NOT pre-loaded: query the metadata service on
          demand only when a task needs it — AWS IMDS at 169.254.169.254 (IMDSv2
          requires a token first: PUT /latest/api/token with
          `X-aws-ec2-metadata-token-ttl-seconds`, then pass it back in
          `X-aws-ec2-metadata-token`), GCP at metadata.google.internal with
          header `Metadata-Flavor: Google`. Treat instance-id, account-id, and
          IAM role as sensitive: fetch only when the task requires it, and do
          not volunteer them.
        - Service scope and logs (systemd hosts). Units are split by manager:
          `running_services` are SYSTEM units (`systemctl status <unit>`,
          `systemctl restart <unit>`, logs via `journalctl -u <unit>`);
          `running_services_user` are PER-USER units started with
          `systemctl --user` (`systemctl --user status <unit>`, logs via
          `journalctl --user -u <unit>`) — these take NO sudo, and sudo'ing
          them queries the wrong manager and fails. Match the scope to the list
          the unit appears in. On a systemd host `journalctl` is the log source
          for a service: prefer `journalctl -u <unit> -n 100 --no-pager`
          (narrow with `--since`, `-p err`, `-b`) over searching `/var/log` for
          a unit's output. If a named service is in NEITHER list, confirm with
          `systemctl status <unit>` then `systemctl --user status <unit>`
          before assuming a log-file path — `running_services_user` is empty
          when the user D-Bus session was unreachable at probe time (sudo,
          cron, no linger), which does not mean the unit is absent.
        - The approval prompt is the permission gate, not you. Do NOT ask the
          user whether to run an inspection/read-only command ("Would you like
          me to check…?") — propose it directly with run_command and let them
          approve or decline. Reply in plain text without a tool call ONLY when
          the task is genuinely finished, or when you need information only the
          user has (a decision, a value, a disambiguation) — never merely to
          ask permission to look.
        - State-modifying commands (writes under /etc, systemctl
          enable/disable, package installs, config drop-ins, and the like)
          are different: propose one only after the user has explicitly
          requested the change or selected it from options you presented.
          A question about whether a change is desirable ("should we…",
          "do we want…") is a request for analysis — answer in plain text
          with a recommendation and the options, then wait for the user's
          choice before proposing the command.
        - For disk SMART/health: a `device` whose `transport` is `usb` sits
          behind a USB bridge, which masks whether the drive is SATA or NVMe,
          so a bare `smartctl -a /dev/sdX` often fails to auto-detect it. When
          the device record carries `smartctl_device_type`, pass it directly:
          `smartctl -d <type> -a /dev/sdX`. If only `usb_ids` is present (an
          unrecognized bridge), try `-d sat` first, then the NVMe pass-through
          types (`-d sntasmedia`, `-d sntjmicron`). smartctl usually needs root.
        - Interpreting NVMe SMART health: the primary signals are SMART overall
          health, Critical Warning, Available Spare vs its threshold,
          Percentage Used, and Media/Data Integrity Errors. Unsafe Shutdowns,
          Warning/Critical Composite Temperature Time, and Error Information Log
          Entries are CUMULATIVE lifetime totals — read them as a rate of
          change across checks, not as alarming absolutes; a large number that
          is no longer increasing is historical residue, not a current fault.
          On a USB-bridged drive the unsafe-shutdown count is especially
          low-signal: many bridges never forward the NVMe shutdown notification
          even on a clean unmount, so it climbs under normal use. Flag a counter
          only when it is actively rising or a threshold (spare, temperature,
          percentage-used) is breached.
        - Never assert that a measurement, sensor, or capability is absent on
          this host without checking first. Negative claims require the same
          verification as positive ones: before saying "there is no way to
          read X here", run the one-line probe that would reveal it (a sysfs
          read, a tool query, a --help). Use the `hardware` facts (model,
          platform_class) to judge which telemetry classes are plausible:
          battery state on a laptop, PMIC/firmware sensors on an SBC, IPMI on
          a server, little to none inside a VM or container.
        - Raspberry Pi hosts expose firmware telemetry via `vcgencmd`:
          `vcgencmd get_throttled` decodes undervoltage/throttling state, and
          on a Pi 5 `vcgencmd pmic_read_adc` reads per-rail PMIC current and
          voltage — a usable live power-draw measurement. When the hardware
          model says Raspberry Pi and `vcgencmd` is in `available_tools`,
          prefer measuring with it over inferring power/thermal state.
        - `network_interfaces` (when present) maps interface name to kind
          (wifi/ethernet/...). Names and roles are stable; use them to pick
          the right interface on turn one instead of probing. IPs, MACs, and
          link state are deliberately NOT included — read live state with
          `ip addr` / `ifconfig <if>` when a task needs it.
        - On Darwin, Wi-Fi identifiers (SSID, BSSID, NetworkID) are
          location-privacy redacted for CLI tools without a location
          entitlement: `system_profiler`, `wdutil info`, and
          `ipconfig getsummary <if>` all print `<redacted>`. Do NOT iterate
          through tools expecting one to escape redaction — none do. If a
          task needs to know WHICH network the host is on, state the
          limitation and ask the user. The redaction covers ONLY those
          identifiers: `ipconfig getsummary <if>` is otherwise the one-shot
          L3 diagnostic on Darwin — IPv4/IPv6 addresses, router, DHCP
          server and lease times, subnet, DNS, link status, and Wi-Fi
          security mode (e.g. WPA3_SAE) in a single unprivileged read.
          Prefer it over chaining `route -n get default` / `netstat -rn` /
          `ifconfig` probes.
        - Keep host data on the host. Do not propose a command that sends
          anything read from this machine (MAC or IP addresses, serial
          numbers, hostnames, usernames, file contents, logs, keys, command
          output) to an external service such as a lookup API, paste site,
          search URL, or webhook, unless the user explicitly asked for that
          transfer. Look it up locally instead. A MAC vendor (OUI) resolves
          offline on systemd hosts with `systemd-hwdb query OUI:XXXXXX` (the
          first six hex digits, uppercase, no separators), and from nmap's
          `nmap-mac-prefixes` or `ieee-data`'s `oui.txt` when installed. If no
          local source exists, say so and let the user decide whether to look
          it up externally. Requests that carry nothing from the host
          (package-index updates, a connectivity check against a public URL,
          fetching documentation) are unaffected.
        - Never propose commands that wipe disks, format filesystems, or
          irrecoverably destroy data. If such a step is genuinely required,
          flag it in plain text and ask the user to run it manually.
        - If a missing software package might help an investigation outcome,
          always state the suggestion and rationale, and await user approval,
          before incorporating a package install into a suggested command. 
    """).strip()


def _norm_cmd(cmd: str) -> str:
    """Whitespace-collapsed command string for cheap exact-match comparison.
    Deliberately NOT semantic: `systemctl status x` and `systemctl status
    x.service` stay distinct. We only ever claim agreement on literal equality,
    never equivalence — overstating sameness is the failure mode to avoid."""
    return " ".join(cmd.split())


def _turn_commands(turn: ChatTurn) -> list[tuple[str, str]]:
    """(command, explanation) pairs a ChatTurn proposed via run_command."""
    out: list[tuple[str, str]] = []
    for tc in turn.tool_calls:
        if tc.name != TOOL_NAME:
            continue
        cmd, why, arg_err = parse_command_args(tc.arguments)
        if arg_err:
            continue
        out.append((cmd, why))
    return out


def _first_move_commands(events: list["CanonicalEvent"]) -> list[tuple[str, str]]:
    """run_command (command, explanation) pairs from the FIRST assistant turn in
    `events` that proposed any — the comparable 'first move'. Shared by both
    /consult paths (normal: tail after the last user msg; aborted: the rolled-
    back turn's captured events)."""
    out: list[tuple[str, str]] = []
    for ev in events:
        if ev.kind == "assistant" and ev.tool_calls:
            for tc in ev.tool_calls:
                if tc.name == TOOL_NAME:
                    cmd, why, arg_err = parse_command_args(tc.arguments)
                    if arg_err:
                        continue
                    out.append((cmd, why))
            break
    return out


def _cmd_count_suffix(n: int) -> str:
    """Dim ' — N command(s)' tag making the broad-batch vs narrow-iterative
    opening explicit. A consult 'first move' is one TURN, which may bundle
    several commands (the API's 'parallel tool calls'). The label avoids the
    word 'parallel' on purpose: in the REPL these are approved AND executed
    one at a time, sequentially — N here means N separate approval prompts you
    would face, not concurrency."""
    if n <= 0:
        return ""
    if n == 1:
        return dim(" — 1 command")
    return dim(f" — {n} commands in its opening turn")


def print_consult(
    active: Provider,
    reference: list[tuple[str, str]],
    results: list[tuple[str, str, Any]],
    ref_aborted: bool = False,
    fresh: bool = False,
) -> None:
    """Render a side-by-side second opinion. `reference` is the active
    provider's most recent proposal (what you'd otherwise run); `results` is a
    list of (provider_name, model, ChatTurn | error_str) from the consulted
    providers. `ref_aborted` marks a reference from a turn you aborted before
    running; `fresh` marks that consulted providers saw only the question, no
    prior history. Executes nothing — this is a read-only decision aid."""
    print()
    print(banner("── consult ─────────────────────────────────────────────────"))
    ref_norm = {_norm_cmd(c) for c, _ in reference}
    if reference:
        moment = ("first move on the aborted question (not run)"
                  if ref_aborted else "first move on this question")
        print(dim(f"reference — {active.name}/{active.model} {moment}")
              + _cmd_count_suffix(len(reference)) + dim(":"))
        for c, why in reference:
            print(f"  {cmd_label(visible(c)[0])}")
            if why:
                print(f"  {dim('— ' + visible(why, newlines=False)[0])}")
    else:
        if ref_aborted:
            print(dim(f"reference — {active.name}/{active.model} was aborted "
                      "before it proposed a command."))
        else:
            print(dim(f"reference — {active.name}/{active.model} answered "
                      "directly (no command run)."))
    below = ("below: how each other provider would START the same question "
             "(read-only — its opening turn; multi-command turns are approved "
             "and run one command at a time).")
    if fresh:
        below += " [--fresh: consulted on the question alone, no prior history]"
    print(dim(below))

    proposed_norms: list[set[str]] = []
    for name, model, turn in results:
        print(dim("─" * 60))
        if isinstance(turn, str):                       # error path
            print(f"{agent_tag(name)} {dim('(' + model + ')')}")
            print(fail(f"  [error] {turn}"))
            continue
        # A refusal or cutoff can leave an unfinished first command.
        cmds = [] if turn.stop else _turn_commands(turn)
        proposed_norms.append({_norm_cmd(c) for c, _ in cmds})
        print(f"{agent_tag(name)} {dim('(' + model + ')')}"
              + _cmd_count_suffix(len(cmds)))
        if turn.stop:
            print(warn(f"  {stop_notice(turn)}"))
        if cmds:
            for c, why in cmds:
                tag = ok("  [matches reference]") if _norm_cmd(c) in ref_norm else ""
                print(f"  {cmd_label(visible(c)[0])}{tag}")
                if why:
                    print(f"  {dim('— ' + visible(why, newlines=False)[0])}")
        if turn.text.strip():
            body = turn.text.strip()
            if len(body) > 600:
                body = body[:600] + dim(" …[truncated]")
            print(textwrap.indent(body, "  "))
        if not cmds and not turn.text.strip():
            print(dim("  (no command, no text)"))

    # Check whether consulted providers proposed exactly the same first command.
    nonempty = [s for s in proposed_norms if s]
    if len(nonempty) >= 2:
        common = set.intersection(*nonempty)
        if common and all(s == nonempty[0] for s in nonempty):
            print(dim("─" * 60))
            print(ok("convergence: consulted providers proposed identical command(s)."))
        elif not common:
            print(dim("─" * 60))
            print(warn("divergence: the providers would start differently — "
                       "compare approaches before trusting one."))
    print(banner("────────────────────────────────────────────────────────────"))
    print(dim("consult is read-only; nothing above was executed."))


def _consult_plan(
    provider_name: str, messages: list[dict],
    aborted_turn_events: list[CanonicalEvent] | None, fresh: bool,
) -> tuple[list[CanonicalEvent], list[tuple[str, str]], bool] | None:
    """What /consult sends and compares: (the transcript each consulted
    provider gets, the active provider's first move as the reference, whether
    that reference is from an aborted turn). None when there is no question.

    Two sources for the question:
     (a) an ABORTED turn: its prompt was rolled out of `messages`, so re-pose
         it. This is the "second opinion before running anything" path.
     (b) normal: the last user message still in `messages`, re-posed so each
         provider works it INDEPENDENTLY from the same prior context rather
         than continuing past the active provider's already-complete answer
         (which leaves nothing to respond to: an empty or redundant turn).
         Read-only, so only each provider's FIRST move is compared.

    The default keeps prior context (apples-to-apples with what the active
    provider saw); `fresh` drops it to the bare question, for a context
    switch where the accumulated session is dead weight paid for on every
    consulted provider. Rollback (q) discards only the aborted turn, not the
    earlier session, so both paths carry the same prior history."""
    aborted_q: str | None = None
    if aborted_turn_events is not None:
        au = next((e for e in aborted_turn_events if e.kind == "user"), None)
        aborted_q = au.text.strip() if au else None

    if aborted_q and aborted_turn_events is not None:
        question = aborted_q
        prior = normalize(provider_name, messages)
        reference = _first_move_commands(aborted_turn_events)
        ref_aborted = True
    else:
        if not any(m.get("role") == "user" and isinstance(m.get("content"), str)
                   for m in messages):
            return None
        events = normalize(provider_name, messages)
        last_user = max(i for i, ev in enumerate(events) if ev.kind == "user")
        question = events[last_user].text
        prior = events[:last_user]
        reference = _first_move_commands(events[last_user + 1:])
        ref_aborted = False

    q_event = CanonicalEvent("user", text=question)
    return ([q_event] if fresh else prior + [q_event]), reference, ref_aborted


def run_consult(
    args: list[str], provider: Provider, messages: list[dict],
    aborted_turn_events: list[CanonicalEvent] | None, system: str,
    thinking: bool, effort: str,
) -> None:
    """/consult [--fresh|fresh|-f]: pose the current (or just-aborted)
    question to every other configured provider at once and print each one's
    opening move beside the active provider's. Read-only: it changes no
    session state and executes nothing. Ctrl-C cancels it and returns."""
    targets = [n for n in available_provider_names() if n != provider.name]
    if not targets:
        print(dim("[no other providers configured — set another API key "
                  "to enable consult]"))
        return
    fresh = any(a in ("--fresh", "fresh", "-f") for a in args)
    plan = _consult_plan(provider.name, messages, aborted_turn_events, fresh)
    if plan is None:
        print(dim("[no conversation yet — ask something first, then "
                  "/consult for a second opinion]"))
        return
    trimmed, reference, ref_aborted = plan

    def consult_one(name: str) -> tuple[str, str, Any]:
        p = make_provider(name)
        p.cache_conversation = False    # one-shot: see Provider
        rendered = render(p, trimmed, system)
        turn = p.chat(rendered, system, thinking=thinking, effort=effort)
        return name, p.model, turn

    # Daemon threads and a queue, not a ThreadPoolExecutor: an executor joins
    # its workers on exit, so Ctrl-C could not return at once and /exit after a
    # cancelled consult would hang for an SDK timeout. Cancelled requests still
    # finish (and bill) in the background; their results are discarded.
    results_q: queue.Queue[tuple[str, str, Any]] = queue.Queue()

    def consult_worker(name: str) -> None:
        try:
            results_q.put(consult_one(name))
        except Exception as e:                  # noqa: BLE001
            results_q.put((name, DEFAULT_MODELS.get(name, "?"),
                           explain_api_error(e)))

    consult_results: list[tuple[str, str, Any]] = []
    try:
        with Activity(f"consulting {len(targets)} provider(s)"):
            for n in targets:
                threading.Thread(target=consult_worker, args=(n,),
                                 name=f"consult-{n}", daemon=True).start()
            for _ in targets:
                consult_results.append(results_q.get())
    except KeyboardInterrupt:
        print()
        print(dim("[consult cancelled; in-flight requests abandoned]"))
        return
    # Stable display order matching available_provider_names().
    order = {n: i for i, n in enumerate(available_provider_names())}
    consult_results.sort(key=lambda r: order.get(r[0], 99))

    print_consult(provider, reference, consult_results,
                  ref_aborted, fresh)

    # Report consult costs separately from the live conversation's token totals.
    c_in = sum(t.usage.input_tokens for _, _, t in consult_results
               if not isinstance(t, str))
    c_out = sum(t.usage.output_tokens for _, _, t in consult_results
                if not isinstance(t, str))
    if c_in or c_out:
        print(dim(f"[consult cost (separate from session): "
                  f"in={c_in} out={c_out}]"))


def run_repl(provider: Provider) -> None:
    global _color_enabled, _audit_path
    facts_verbose = False
    facts = gather_host_facts()
    system = build_system_prompt(facts)
    messages = provider.initial_messages(system)
    auto_approve = False
    show_tokens = False                  # /tokens on|off
    thinking_enabled = ANTHROPIC_THINKING_DEFAULT   # /thinking on|off
    effort = ANTHROPIC_THINKING_EFFORT_DEFAULT       # /effort low..max (adaptive)
    session_in = 0                       # cumulative prompt tokens (cached incl.)
    session_out = 0                      # cumulative output tokens this session
    last_usage = Usage()                 # token usage from most recent call
    ctx_window = CONTEXT_WINDOWS.get(provider.model)   # may be None
    # Keep the last aborted question for /consult, even after a provider
    # switch. Clear it after the next normally completed turn.
    aborted_turn_events: list[CanonicalEvent] | None = None

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

    def audit(action: str, command: str, *, explanation: str = "",
              edited: str | None = None, returncode: int | None = None,
              truncated: bool | None = None, reason: str | None = None,
              note: str | None = None,
              stdout: str | None = None, stderr: str | None = None) -> None:
        """Stamp the current host/provider/model onto a record and append it to
        the audit log. Reads provider/facts live, so a mid-session /provider,
        /model, or /facts switch is reflected. No-op when auditing is off."""
        if _audit_path is None:
            return
        rec: dict[str, Any] = {
            "ts": datetime.datetime.now(datetime.timezone.utc)
                  .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": facts.get("node", ""),
            "provider": provider.name,
            "model": provider.model,
            "action": action,
            "command": command,
        }
        if explanation:
            rec["explanation"] = explanation
        if edited is not None:
            rec["edited_command"] = edited
        if returncode is not None:
            rec["returncode"] = returncode
        if truncated is not None:
            rec["truncated"] = truncated
        if reason is not None:
            rec["reason"] = reason
        if note is not None:
            rec["note"] = note
        if _audit_body:
            if stdout is not None:
                rec["stdout"] = stdout[:OUTPUT_MAX_CHARS]
            if stderr is not None:
                rec["stderr"] = stderr[:OUTPUT_MAX_CHARS]
        write_audit(rec)

    if thinking_enabled or provider.thinking_active(thinking_enabled):
        thinking_note = f"  thinking={provider.thinking_status(thinking_enabled)}"
        if (thinking_enabled and provider.thinking_active(thinking_enabled)
                and provider.effort_applies()):
            thinking_note += f" effort={effort}"
    else:
        thinking_note = ""
    print(banner(
        f"sys_agent  provider={provider.name}  model={provider.model}  "
        f"host={facts['node']} ({facts['system']}/{facts['machine']})"
        f"{thinking_note}"
    ))
    print(dim("meta: " + "  ".join(BANNER_META_COMMANDS)
              + "  …  /help for all"))
    if _update_check and (notice := _update_check.take(UPDATE_BANNER_WAIT)):
        print(warn(f"[{notice}]"))
    if SHOW_DISCLAIMER:
        print()
        print(warn("note:") + " " + dim(DISCLAIMER))
    print()

    while True:
        if _update_check and (notice := _update_check.take()):
            print(warn(f"[{notice}]"))
        try:
            # The uppercase tag tells this apart from a shell prompt. Cyan
            # matches the approval prompt (waiting on you); magenta is used
            # nowhere else, so the host stands out, and neither is the Pi
            # PS1's green. colored_input() handles prompt width per backend.
            prompt = (user_tag("SYS_AGENT") + host_tag(f"@{facts['node']}")
                      + user_tag(">") + " ")
            user_in = colored_input(prompt).strip()
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

        pasted = user_in.startswith(PASTE_DELIM)
        if pasted:
            try:
                user_in = read_paste_block(user_in)
            except KeyboardInterrupt:
                print()
                print(dim("[multi-line input cancelled]"))
                continue
            if not user_in:
                continue

        # Add history explicitly for consistent behavior with colored prompts.
        # Drop meta-commands and store pasted text on one line for recall.
        if _HAVE_READLINE:
            try:
                readline.add_history(" ".join(user_in.split()) if pasted
                                     else user_in)
            except Exception:   # noqa: BLE001
                pass
        if user_in.startswith("/") and not pasted:
            drop_last_history_entry()

        # Treat every pasted block as conversation, even if it starts with "/".
        meta = "" if pasted else user_in
        if meta in ("/exit", "/quit"):
            return
        if meta in ("/help", "/?"):
            print_help(provider, auto_approve, show_tokens, thinking_enabled, effort)
            continue
        if meta == "/reset":
            messages = provider.initial_messages(system)
            session_in = session_out = 0
            last_usage = Usage()
            # Remove the saved question too so /consult respects the reset.
            aborted_turn_events = None
            print(dim("[conversation reset, token counters cleared]"))
            continue
        if meta == "/version":
            # Let Ctrl-C cancel an in-progress update check.
            try:
                with Activity("checking GitHub"):
                    lines = version_report()
            except KeyboardInterrupt:
                print()
                print(dim("[version check cancelled]"))
                continue
            print("\n".join(lines))
            continue
        if meta == "/info":
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
                "thinking_enabled": thinking_enabled,
                "thinking_active": provider.thinking_active(thinking_enabled),
                "thinking_mode": provider.thinking_mode(),
                "effort": effort,
                "color": _color_enabled,
                "audit_enabled": _audit_path is not None,
                "audit_path": _audit_path,
                "audit_body": _audit_body,
                "host": facts,
            }, indent=2))
            continue
        if meta.startswith("/provider"):
            parts = meta.split()
            if len(parts) == 1:
                print(json.dumps({
                    "current_provider": provider.name,
                    "current_model": provider.model,
                    "available_providers": available_provider_names(),
                }, indent=2))
                continue
            if len(parts) != 2 or parts[1].lower() not in (
                "openai", "anthropic", "deepseek", "o", "a", "claude", "d", "ds"
            ):
                print(dim("usage: /provider openai|anthropic|deepseek"))
                continue
            # Switch only after the new SDK client is ready so Ctrl-C can cancel.
            try:
                provider = make_provider(parts[1])
            except ValueError as e:
                print(err_bold(f"[provider switch failed] {e}"))
                continue
            except KeyboardInterrupt:
                print()
                print(dim("[provider switch cancelled]"))
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
        if meta.startswith("/model"):
            parts = meta.split(maxsplit=1)
            requested = parts[1].strip() if len(parts) == 2 else ""
            known = PROVIDER_MODELS.get(provider.name, ())
            prefixes = PROVIDER_MODEL_PREFIXES.get(provider.name, ())
            if requested and requested in known:
                chosen = requested
            elif requested and requested.startswith(prefixes):
                # Accept a matching model prefix with a warning.
                print(warn(unlisted_model_warning(provider.name, requested)))
                chosen = requested
            else:
                # Offer the picker for a missing or unrecognized name.
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
        if meta.startswith("/facts"):
            parts = meta.split()
            if len(parts) == 1:
                print(json.dumps(facts, indent=2))
                continue
            refresh = len(parts) == 2 and parts[1] == "refresh"
            verbose = (len(parts) == 3 and parts[1] == "verbose"
                       and parts[2] in ("on", "off"))
            if refresh or verbose:
                want_verbose = parts[2] == "on" if verbose else facts_verbose
                # Apply the new facts only after the probe has finished.
                try:
                    new_facts = (gather_verbose_host_facts() if want_verbose
                                 else gather_host_facts())
                    new_system = build_system_prompt(new_facts)
                except KeyboardInterrupt:
                    print()
                    print(dim("[facts refresh cancelled; previous facts kept]"))
                    continue
                facts_verbose, facts, system = want_verbose, new_facts, new_system
                messages = provider.initial_messages(system)
                session_in = session_out = 0
                last_usage = Usage()
                print(dim(
                    f"[facts verbose = {facts_verbose}; host facts refreshed; "
                    "conversation/token counters reset]" if verbose else
                    "[host facts refreshed; conversation/token counters reset]"
                ))
                continue
            print(dim("usage: /facts | /facts refresh | /facts verbose on|off"))
            continue
        if meta.startswith("/auto"):
            parts = meta.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                auto_approve = parts[1] == "on"
                print(dim(f"[auto-approve = {auto_approve}]"))
            else:
                print(dim(f"[auto-approve = {auto_approve}]  usage: /auto on|off"))
            continue
        if meta.startswith("/thinking"):
            parts = meta.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                thinking_enabled = parts[1] == "on"
                print(dim(
                    f"[thinking = {provider.thinking_status(thinking_enabled)}]"
                ))
            else:
                print(dim(
                    f"[thinking = {provider.thinking_status(thinking_enabled)}]"
                    "  usage: /thinking on|off"
                ))
            continue
        if meta.startswith("/effort"):
            parts = meta.split()
            # Show the same effective effort when setting or inspecting it.
            if len(parts) == 2 and parts[1].lower() in _VALID_EFFORTS:
                effort = parts[1].lower()
                with_usage = False
            else:
                with_usage = True
            shown, note = effort_display(provider, effort, thinking_enabled)
            note_str = f"  ({note})" if note else ""
            usage = (f"  usage: /effort {'|'.join(_effort_usage_levels(provider))}"
                     if with_usage else "")
            print(dim(f"[effort = {shown}]{note_str}{usage}"))
            continue
        if meta.startswith("/tokens"):
            parts = meta.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                show_tokens = parts[1] == "on"
                print(dim(f"[show_tokens = {show_tokens}]"))
            else:
                # Bare /tokens prints current snapshot regardless of toggle
                print(fmt_tokens(last_usage))
                print(dim(f"  show_tokens = {show_tokens}  (usage: /tokens on|off)"))
            continue
        if meta.startswith("/color"):
            parts = meta.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                _color_enabled = parts[1] == "on"
                print(dim(f"[color = {_color_enabled}]"))
            else:
                print(dim(f"[color = {_color_enabled}]  usage: /color on|off"))
            continue
        if meta.startswith("/audit"):
            parts = meta.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                if parts[1] == "off":
                    _audit_path = None
                    print(dim("[audit = off]"))
                else:
                    raw = os.environ.get("SYS_AUDIT_LOG", AUDIT_LOG_DEFAULT)
                    if raw.strip().lower() in _AUDIT_OFF_VALUES:
                        raw = AUDIT_LOG_DEFAULT
                    _audit_path = _resolve_audit_path(raw)
                    print(dim(
                        f"[audit = on  path={_audit_path}]" if _audit_path
                        else "[audit = off (path unavailable)]"
                    ))
            elif len(parts) == 1:
                if _audit_path:
                    print(dim(
                        f"[audit on  path={_audit_path}  body={_audit_body}]"
                    ))
                else:
                    print(dim("[audit off]"))
            else:
                print(dim(
                    f"[audit {'on' if _audit_path else 'off'}]  "
                    "usage: /audit | /audit on|off"
                ))
            continue
        if meta.startswith("/history"):
            parts = meta.split()
            limit: int | None = 50
            if len(parts) == 2 and parts[1] == "all":
                limit = None
            elif len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0:
                limit = int(parts[1])
            elif len(parts) >= 2:
                print(dim("usage: /history | /history N | /history all"))
                continue
            hist_path = _audit_path or os.path.expanduser(AUDIT_LOG_DEFAULT)
            # Let Ctrl-C cancel slow log reads or waits for the pager to exit.
            try:
                text = render_history(hist_path, limit)
                if text is None:
                    print(dim("[no audit history yet]"))
                else:
                    page_text(text)
            except KeyboardInterrupt:
                print()
                print(dim("[history cancelled]"))
            continue
        if meta.startswith("/consult"):
            run_consult(meta.split()[1:], provider, messages,
                        aborted_turn_events, system, thinking_enabled, effort)
            continue

        # Remember where this turn starts so cancellation can remove it from history.
        turn_start = len(messages)
        messages.append({"role": "user", "content": user_in})

        # Keep calling the model until it stops requesting tools.
        iteration = 0
        while True:
            act_label = ("thinking"
                         if provider.thinking_active(thinking_enabled)
                         else "working")
            try:
                with Activity(act_label):
                    turn = provider.chat(
                        messages, system,
                        thinking=thinking_enabled, effort=effort,
                    )
            except KeyboardInterrupt:
                # Save the interrupted exchange for /consult and remove it
                # from the live conversation.
                print(fail("\n[interrupted — stopping workflow]"))
                aborted_turn_events = normalize(provider.name,
                                                messages[turn_start:])
                del messages[turn_start:]
                break
            except Exception as e:          # noqa: BLE001
                print(err_bold(f"[api error] {explain_api_error(e)}"))
                # Drop the unanswered question if the very first call failed.
                if iteration == 0:
                    messages.pop()
                break
            iteration += 1

            last_usage = turn.usage
            # Include cached tokens in the session total. The per-call
            # display shows them separately because they bill differently.
            session_in += last_usage.context_tokens
            session_out += last_usage.output_tokens
            if show_tokens:
                print(fmt_tokens(last_usage))

            # An unfinished command can look valid. Check why the model
            # stopped before storing the reply or asking for approval.
            if turn.stop == STOP_REFUSAL:
                # Remove the refused turn from history and keep it for /consult.
                print(fail(f"\n{stop_notice(turn)}"))
                if turn.tool_calls:
                    print(fail("[proposed command discarded — nothing ran]"))
                aborted_turn_events = normalize(provider.name,
                                                messages[turn_start:])
                del messages[turn_start:]
                break
            if turn.stop == STOP_TRUNCATED and turn.tool_calls:
                # Drop the incomplete command. Remove the user's question on
                # the first call. On later calls, keep earlier tool results
                # so the conversation can continue from there.
                print(warn(f"\n{stop_notice(turn)}"))
                print(warn("[proposed command discarded as incomplete — "
                           "nothing ran]"))
                if iteration == 1:
                    messages.pop()
                break

            provider.append_assistant(messages, turn)

            if turn.thinking_text:
                print(dim(textwrap.indent(turn.thinking_text, "  │ ")))

            if not turn.tool_calls:
                if turn.text:
                    # Colored prefix, plain body (user preference).
                    print(f"\n{dim(f'[{short_model(provider.model)}]')} "
                          f"{agent_tag('agent>')} {turn.text}\n")
                if turn.stop == STOP_TRUNCATED:
                    # Keep partial text in history and mark it as cut off.
                    print(warn(stop_notice(turn)))
                # /consult now uses the question from this completed turn.
                aborted_turn_events = None
                break

            results: list[tuple[str, str]] = []
            stop_turn = False
            for tc in turn.tool_calls:
                if tc.name != TOOL_NAME:
                    results.append((tc.id, json.dumps({"error": f"unknown tool: {tc.name}"})))
                    continue

                cmd, why, arg_err = parse_command_args(tc.arguments)
                if arg_err:
                    results.append((tc.id, json.dumps({"error": arg_err})))
                    continue
                why = why or "(no explanation)"

                deny = is_denied(cmd)
                if deny:
                    print(f"\n{err_bold('[blocked]')} {visible(cmd)[0]}")
                    print(fail(f"   reason: {deny}"))
                    results.append((tc.id, json.dumps({"error": f"blocked locally: {deny}"})))
                    audit("deny", cmd, explanation=why, reason=deny)
                    continue

                action, edited = prompt_approval(cmd, why, auto_approve)
                if action == "stop":
                    # Stop before running any remaining tools in this turn.
                    audit("abort", cmd, explanation=why)
                    stop_turn = True
                    break
                if action == "skip":
                    results.append((tc.id, json.dumps({"error": "user declined to run command"})))
                    audit("skip", cmd, explanation=why)
                    continue

                to_run = edited or cmd
                if edited:
                    # Recheck the edited command because it replaces the
                    # proposal that already passed the deny check.
                    deny = is_denied(to_run)
                    if deny:
                        print(f"\n{err_bold('[blocked]')} {visible(to_run)[0]}")
                        print(fail(f"   reason: {deny}"))
                        results.append((tc.id, json.dumps(
                            {"error": f"blocked locally (edited command): {deny}"})))
                        audit("deny", cmd, explanation=why,
                              edited=to_run, reason=deny)
                        continue
                    print(f"{warn('[running edited]:')} {visible(to_run)[0]}")

                try:
                    with Activity("running"):
                        result = execute(to_run)
                except KeyboardInterrupt:
                    # execute() has already stopped the process group.
                    # Record the interrupt before rolling back the conversation.
                    print(fail("\n[interrupted — command cancelled]"))
                    audit("edit" if edited else "run", cmd,
                          explanation=why, edited=edited,
                          note="interrupted (workflow stopped)")
                    stop_turn = True
                    break
                exit_str = f"[exit={result.returncode}]"
                print(ok(exit_str) if result.returncode == 0 else fail(exit_str))
                if result.stdout:
                    # stdout preserved raw — may contain its own ANSI
                    print(result.stdout.rstrip())
                if result.stderr:
                    print(warn("[stderr]"))
                    print(result.stderr.rstrip())

                payload = result.as_tool_payload()   # sets result.truncated
                results.append((tc.id, payload))
                audit("edit" if edited else "run", cmd, explanation=why,
                      edited=edited, returncode=result.returncode,
                      truncated=result.truncated,
                      stdout=result.stdout, stderr=result.stderr)

            if stop_turn:
                # Remove the interrupted turn from the live conversation.
                # Commands already run keep their effects and audit records.
                aborted_turn_events = normalize(provider.name,
                                                messages[turn_start:])
                del messages[turn_start:]
                print(fail("[workflow stopped — back to prompt]"))
                break

            provider.append_tool_results(messages, results)


# -----------------------------------------------------------------------------
# Startup update check
# -----------------------------------------------------------------------------
# Compare the running file with GitHub main by content hash. When the file
# differs and is in a checkout, ask GitHub how the commits compare before
# showing a notice. Report differences for standalone copies. Fail quietly
# so offline hosts and API rate limits do not interrupt startup.

def _git_blob_sha(data: bytes) -> str:
    """The object id `git hash-object` gives this content; GitHub's contents
    API reports the same value as `sha`."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _github_json(path: str) -> Any:
    req = urllib.request.Request(
        UPDATE_REPO_API + path,
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "sys_agent-update-check"})
    with urllib.request.urlopen(req, timeout=UPDATE_HTTP_TIMEOUT) as resp:
        return json.load(resp)


def _git_head(directory: str) -> str | None:
    """HEAD commit of the checkout holding this file, or None when it is not
    a checkout or git is unavailable."""
    head = (_probe(["git", "-C", directory, "rev-parse", "HEAD"],
                   timeout=UPDATE_HTTP_TIMEOUT) or "").strip()
    return head if re.fullmatch(r"[0-9a-f]{40}", head) else None


def update_status(path: str | None = None) -> tuple[bool, str]:
    """(notify, message) for the running file against GitHub main. notify is
    True only when the user may be missing main's code; the message is
    always set, so /version can report the silent outcomes too (current,
    ahead, local edits, or why the check failed). Never raises Exception."""
    try:
        # Follow the launcher symlink to find the actual Git checkout.
        path = os.path.realpath(path or __file__)
        with open(path, "rb") as fh:
            local = _git_blob_sha(fh.read())
        remote = _github_json(f"/contents/sys_agent.py?ref={UPDATE_BRANCH}")
        if remote.get("sha") == local:
            return False, f"up to date with GitHub {UPDATE_BRANCH}"
        directory = os.path.dirname(path)
        head = _git_head(directory)
        if head is None:
            return True, (f"sys_agent.py differs from GitHub {UPDATE_BRANCH} "
                          "(an update, or local edits)")
        try:
            cmp = _github_json(f"/compare/{head}...{UPDATE_BRANCH}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            # GitHub cannot compare against a commit it does not have.
            return True, (f"sys_agent.py differs from GitHub {UPDATE_BRANCH} and "
                          "this checkout has commits GitHub does not")
        # In this comparison, "ahead" means GitHub main has newer commits.
        status = cmp.get("status")
        behind, ahead = cmp.get("ahead_by", 0), cmp.get("behind_by", 0)
        if status == "ahead":
            return True, (f"update available: {behind} commit{'s' * (behind != 1)} "
                          f"behind GitHub {UPDATE_BRANCH} — run: git -C {directory} pull")
        if status == "diverged":
            return True, (f"this checkout has diverged from GitHub {UPDATE_BRANCH} "
                          f"({behind} behind, {ahead} ahead)")
        if status == "behind":
            return False, (f"{ahead} commit{'s' * (ahead != 1)} ahead of GitHub "
                           f"{UPDATE_BRANCH} (newer than main)")
        if status == "identical":
            return False, (f"at GitHub {UPDATE_BRANCH}'s commit, with local "
                           "edits to sys_agent.py")
        return False, f"could not check GitHub: unexpected compare status {status!r}"
    except urllib.error.HTTPError as e:
        hint = " (rate limited?)" if e.code in (403, 429) else ""
        return False, f"could not check GitHub: HTTP {e.code}{hint}"
    except urllib.error.URLError as e:
        return False, f"could not check GitHub: {e.reason}"
    except Exception as e:                                    # noqa: BLE001
        return False, f"could not check GitHub: {type(e).__name__}: {e}"


def check_for_update(path: str | None = None) -> str | None:
    """Startup notice: the message only when the user may be missing main's
    code. Everything else, failures included, is silent at startup."""
    notify, message = update_status(path)
    return message if notify else None


def _git_describe(directory: str) -> str | None:
    """`git describe --tags --dirty --always` for the checkout holding this
    file: release tag, commits past it, commit, and local edits in one."""
    out = (_probe(["git", "-C", directory, "describe", "--tags", "--dirty",
                   "--always"], timeout=UPDATE_HTTP_TIMEOUT) or "").strip()
    return out or None


def _readline_backend() -> str:
    """Live line-editing backend. GNU vs libedit is the Pi/Mac asymmetry that
    real bugs have hinged on, so a bug report needs it."""
    if not _HAVE_READLINE:
        return "none"
    if _IS_LIBEDIT:
        return "libedit (stdlib)"
    ver = getattr(readline, "_READLINE_LIBRARY_VERSION", "?")
    src = "gnureadline" if readline.__name__ == "gnureadline" else "stdlib"
    return f"GNU {ver} ({src})"


def _dist_version(name: str) -> str:
    # Read SDK versions without importing the packages.
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def version_report(path: str | None = None) -> list[str]:
    """Lines for /version: which file runs, which version it is, whether it
    matches GitHub main, and the runtime a bug report needs. The GitHub part
    is checked fresh and, unlike the startup check, says so when current and
    reports failures instead of hiding them."""
    invoked = os.path.abspath(path or __file__)
    path = os.path.realpath(invoked)       # through the ~/.local/bin symlink
    via = f" via {invoked}" if invoked != path else ""
    version = _git_describe(os.path.dirname(path))
    if version is None:
        try:
            with open(path, "rb") as fh:
                version = f"unversioned copy (blob {_git_blob_sha(fh.read())[:7]})"
        except OSError as e:
            version = f"unreadable ({e.strerror})"
    _, status = update_status(path)
    return [
        f"sys_agent {version}   ({path}{via})",
        f"github {UPDATE_BRANCH}: {status}",
        f"python {platform.python_version()}   readline: {_readline_backend()}   "
        f"anthropic {_dist_version('anthropic')}   openai {_dist_version('openai')}",
    ]


class UpdateCheck:
    """Runs check_for_update() on a daemon thread from the top of main(), so
    it overlaps provider selection and fact gathering. Daemon, not an
    executor: a request stuck in its timeout must never delay exit."""

    def __init__(self) -> None:
        self._notice: str | None = None
        self._shown = False
        self._thread = threading.Thread(
            target=self._run, name="update-check", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._notice = check_for_update()

    def take(self, wait: float = 0.0) -> str | None:
        """The notice, at most once. Called at the banner with a short wait,
        then before each prompt with none, so a slow network still reports
        (between prompts, never over readline's line)."""
        if self._shown:
            return None
        self._thread.join(wait)
        if self._thread.is_alive():
            return None
        self._shown = True
        return self._notice


_update_check: UpdateCheck | None = None


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def _env_str(name: str, current: str) -> str:
    """Env override for a free-form string. Empty/whitespace means unset."""
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else current


def _env_int(name: str, current: int, *, minimum: int = 1) -> tuple[int, str | None]:
    """Env override for an integer -> (value, problem). Never raises: these are
    resolved before the REPL exists, and a bare int() here used to abort
    startup with a traceback on a typo'd value."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return current, None
    try:
        val = int(raw.strip())
    except ValueError:
        return current, f"{name}={raw.strip()!r} is not an integer — keeping {current}"
    if val < minimum:
        return current, f"{name}={val} is below the minimum {minimum} — keeping {current}"
    return val, None


def _env_flag(name: str, current: bool) -> tuple[bool, str | None]:
    """Env override for an on/off switch -> (value, problem)."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return current, None
    val = raw.strip().lower()
    if val in ("on", "1", "true", "yes"):
        return True, None
    if val in ("off", "0", "false", "no"):
        return False, None
    state = "on" if current else "off"
    return current, f"{name}={raw.strip()!r} is not on/off — keeping {state}"


def _env_choice(name: str, current: str,
                valid: tuple[str, ...]) -> tuple[str, str | None]:
    """Env override constrained to a fixed set -> (value, problem)."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return current, None
    val = raw.strip().lower()
    if val not in valid:
        return current, (f"{name}={raw.strip()!r} is not one of "
                         f"{'|'.join(valid)} — keeping {current}")
    return val, None


def init_config() -> list[str]:
    """Resolve every SYS_* setting from os.environ into the module globals.

    Called once at import (shell environment only) and again from main() after
    the env file has been merged in. Without that second pass every setting
    documented as .env-configurable was captured before the file was ever read,
    so the file's value was silently ignored — the shell path worked, the
    documented file path did not.

    Shell variables still win: load_env_file() never overwrites an existing
    os.environ entry, so re-resolving here cannot demote them.

    Each global keeps its literal default at its definition site above (see the
    `override:` comments); the lookups below only replace it when the variable
    is set, which keeps one source of truth per setting. Idempotent.

    Returns human-readable problems for the caller to print. A bad value is
    reported and ignored, never fatal.
    """
    global DEFAULT_OPENAI_MODEL, DEFAULT_ANTHROPIC_MODEL, DEFAULT_DEEPSEEK_MODEL
    global COMMAND_TIMEOUT, ANTHROPIC_THINKING_DEFAULT
    global ANTHROPIC_THINKING_EFFORT_DEFAULT, ANTHROPIC_THINKING_BUDGET
    global ANTHROPIC_THINKING_MAX_TOKENS, SHOW_DISCLAIMER, SHOW_PROGRESS
    global RUNTIME_TOP_PROCESSES, UPDATE_CHECK

    problems: list[str] = []

    def take(resolved: tuple[Any, str | None]) -> Any:
        value, problem = resolved
        if problem:
            problems.append(problem)
        return value

    DEFAULT_OPENAI_MODEL = _env_str("SYS_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    DEFAULT_ANTHROPIC_MODEL = _env_str("SYS_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    DEFAULT_DEEPSEEK_MODEL = _env_str("SYS_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    # Update in place so existing references see the new defaults.
    # SYS_*_MODEL can pin models outside the picker.
    DEFAULT_MODELS.update({
        "openai": DEFAULT_OPENAI_MODEL,
        "anthropic": DEFAULT_ANTHROPIC_MODEL,
        "deepseek": DEFAULT_DEEPSEEK_MODEL,
    })

    COMMAND_TIMEOUT = take(_env_int("SYS_COMMAND_TIMEOUT", COMMAND_TIMEOUT))
    ANTHROPIC_THINKING_DEFAULT = take(
        _env_flag("SYS_THINKING", ANTHROPIC_THINKING_DEFAULT))
    ANTHROPIC_THINKING_EFFORT_DEFAULT = take(
        _env_choice("SYS_THINKING_EFFORT",
                    ANTHROPIC_THINKING_EFFORT_DEFAULT, _VALID_EFFORTS))
    ANTHROPIC_THINKING_BUDGET = take(
        _env_int("SYS_THINKING_BUDGET", ANTHROPIC_THINKING_BUDGET))
    ANTHROPIC_THINKING_MAX_TOKENS = take(
        _env_int("SYS_THINKING_MAX_TOKENS", ANTHROPIC_THINKING_MAX_TOKENS))
    SHOW_DISCLAIMER = take(_env_flag("SYS_DISCLAIMER", SHOW_DISCLAIMER))
    SHOW_PROGRESS = take(_env_flag("SYS_PROGRESS", SHOW_PROGRESS))
    UPDATE_CHECK = take(_env_flag("SYS_UPDATE_CHECK", UPDATE_CHECK))
    RUNTIME_TOP_PROCESSES = take(
        _env_int("SYS_TOP_PROCESSES", RUNTIME_TOP_PROCESSES))
    return problems


# Shell environment only — main() re-runs this once the env file is loaded.
init_config()


def main() -> None:
    global _update_check
    explicit = os.environ.get("SYS_ENV_FILE")
    env_file = find_env_file(explicit)
    # Load the env file before reading configuration or printing colored output.
    loaded = load_env_file(env_file) if env_file else 0
    init_color()
    init_readline()
    atexit.register(save_readline_history)
    if loaded:
        print(dim(f"[loaded {loaded} vars from {env_file}]"))
    elif explicit and not env_file:
        print(warn(f"[SYS_ENV_FILE={explicit} not found; relying on shell env]"))
    for problem in init_config():
        print(warn(f"[config] {problem}"))
    if UPDATE_CHECK:
        _update_check = UpdateCheck()
    init_audit()
    if _audit_path:
        print(dim(f"[audit log → {_audit_path}]"))
    provider = select_provider()
    run_repl(provider)


if __name__ == "__main__":
    main()
