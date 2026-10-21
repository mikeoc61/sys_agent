#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#     "openai>=3.19",
#     "anthropic>=1.8,<2.0",
#     "gnureadline>=8.1; sys_platform != 'win32'",
# ]
# ///
"""
live_probe.py — live Anthropic API contract probe for sys_agent thinking routing.

Drives the real AnthropicProvider.chat() tool loop against the API. Proposed
commands are NEVER executed: every run_command call is answered with a canned
tool result, so nothing touches the host and the approval gate is not bypassed.

Checks:
  sonnet55_off     claude-sonnet-5-5, /thinking off: every request carries
                   thinking=between_tools via extra_body, no output_config, the
                   4096 non-streaming path; multi-turn tool replay with no 400.
  sonnet55_toggle  claude-sonnet-5-5 on(xhigh) -> off -> on(xhigh) over one
                   growing history: no 400 at any mode switch.
  haiku55_off      claude-haiku-5-5, /thinking off: thinking=disabled, no effort.
  haiku55_on       claude-haiku-5-5, /thinking on (high): adaptive, streamed.
  haiku45_off      claude-haiku-4-5-20251001 baseline for token comparison.
  ds_flash_on      deepseek-flash, /thinking on (high): enabled+reasoning_effort
                   on every request; reasoning_content replays with no 400.
  ds_flash_off     deepseek-flash, /thinking off: thinking=disabled on every
                   request, no reasoning_effort.
  ds_flash_toggle  deepseek-flash on(max) -> off -> on(low) over one history.
  ds_legacy_toggle deepseek-v4-flash (retired name): same toggle; response
                   `model` must show the server-routed deepseek-flash.
DeepSeek checks need DEEPSEEK_API_KEY and record each response's `model`:
DeepSeek's docs say deepseek-flash is served by V4.1-Flash-1017 but the API
reports only the alias, so the snapshot itself cannot be verified here.

Usage: `uv run tools/live_probe.py [check ...]` from the repo root.
Writes ~/.local/state/sys_agent/live_probe_results.json (or under $XDG_STATE_HOME). Exit 0 only if every check passes.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
import sys_agent as S

MAX_ITER = 10

PROMPTS = [
    "Check how full the root filesystem is and how long the system has been up, then summarize in two sentences.",
    "Now check memory usage and tell me whether anything looks concerning.",
    "Last one: what kernel version is running?",
]

CANNED = {
    "df": "Filesystem      Size  Used Avail Use% Mounted on\n/dev/nvme0n1p2  469G  121G  329G  27% /\n",
    "uptime": " 09:50:12 up 41 days,  3:12,  1 user,  load average: 0.21, 0.18, 0.15\n",
    "free": "               total        used        free      shared  buff/cache   available\nMem:           8.0Gi       2.1Gi       3.4Gi        92Mi       2.6Gi       5.7Gi\nSwap:          2.0Gi          0B       2.0Gi\n",
    "uname": "Linux pibot 6.12.47+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.12.47-1+rpt1 aarch64 GNU/Linux\n",
    "cat /proc/meminfo": "MemTotal:        8245680 kB\nMemFree:         3565432 kB\nMemAvailable:    5977104 kB\n",
}


def canned_result(command: str) -> str:
    for key, out in CANNED.items():
        if key in command:
            return S.CmdResult(command, 0, out, "").as_tool_payload()
    return S.CmdResult(command, 0, "(probe: no canned output for this command; answer from the output you already have)\n", "").as_tool_payload()


class Spy:
    """Wrap client.messages.create/stream to record each request's shape."""

    def __init__(self, provider: S.AnthropicProvider) -> None:
        self.requests: list[dict[str, Any]] = []
        msgs = provider.client.messages
        real_create, real_stream = msgs.create, msgs.stream

        def create(**kw: Any) -> Any:
            self._record(kw, streamed=False)
            return real_create(**kw)

        def stream(**kw: Any) -> Any:
            self._record(kw, streamed=True)
            return real_stream(**kw)

        msgs.create = create
        msgs.stream = stream

    def _record(self, kw: dict[str, Any], streamed: bool) -> None:
        eb = kw.get("extra_body") or {}
        self.requests.append({
            "thinking": eb.get("thinking", kw.get("thinking")),
            "output_config": eb.get("output_config"),
            "max_tokens": kw.get("max_tokens"),
            "streamed": streamed,
            "n_messages": len(kw.get("messages", [])),
        })


class DSSpy:
    """Wrap client.chat.completions.create to record extra_body and response model."""

    def __init__(self, provider: S.DeepSeekProvider) -> None:
        self.requests: list[dict[str, Any]] = []
        self.models: list[str] = []
        comp = provider.client.chat.completions
        real = comp.create

        def create(**kw: Any) -> Any:
            eb = kw.get("extra_body") or {}
            self.requests.append({
                "thinking": eb.get("thinking"),
                "reasoning_effort": eb.get("reasoning_effort"),
                "n_messages": len(kw.get("messages", [])),
            })
            resp = real(**kw)
            self.models.append(resp.model)
            return resp

        comp.create = create


def run_prompt(provider: S.Provider, messages: list[dict], system: str,
               prompt: str, thinking: bool, effort: str, log: list[dict]) -> None:
    messages.append({"role": "user", "content": prompt})
    for _ in range(MAX_ITER):
        t0 = time.monotonic()
        turn = provider.chat(messages, system, thinking=thinking, effort=effort)
        dt = time.monotonic() - t0
        if provider.name == "deepseek":
            # content is a string; the scratchpad rides in reasoning_content
            thinking_blocks = [{"thinking": turn.raw_message.get("reasoning_content", "")}] \
                if "reasoning_content" in turn.raw_message else []
        else:
            thinking_blocks = [b for b in turn.raw_message["content"] if b.get("type") == "thinking"]
        log.append({
            "thinking_flag": thinking,
            "effort": effort,
            "stop": turn.stop,
            "stop_detail": turn.stop_detail,
            "tool_calls": [tc.arguments.get("command", tc.name) for tc in turn.tool_calls],
            "thinking_blocks": len(thinking_blocks),
            "thinking_blocks_nonempty": sum(1 for b in thinking_blocks if b.get("thinking")),
            "thinking_text_sample": turn.thinking_text[:300],
            "text_sample": turn.text[:300],
            "status": S.thinking_status(provider, thinking),
            "in": turn.usage.input_tokens,
            "out": turn.usage.output_tokens,
            "cache_read": turn.usage.cache_read,
            "cache_write": turn.usage.cache_write,
            "secs": round(dt, 2),
        })
        if turn.stop in (S.STOP_REFUSAL, S.STOP_TRUNCATED):
            raise RuntimeError(f"stop={turn.stop} {turn.stop_detail}")
        provider.append_assistant(messages, turn)
        if not turn.tool_calls:
            return
        provider.append_tool_results(
            messages,
            [(tc.id, canned_result(str(tc.arguments.get("command", "")))) for tc in turn.tool_calls],
        )
    raise RuntimeError(f"no final answer within {MAX_ITER} model calls")


def session(model: str, phases: list[tuple[bool, str]], system: str) -> dict[str, Any]:
    provider: S.Provider = S.DeepSeekProvider(model) if model.startswith("deepseek") \
        else S.AnthropicProvider(model)
    spy: Spy | DSSpy = DSSpy(provider) if model.startswith("deepseek") else Spy(provider)
    messages: list[dict] = provider.initial_messages(system)
    log: list[dict] = []
    error = None
    try:
        for prompt, (thinking, effort) in zip(PROMPTS, phases):
            run_prompt(provider, messages, system, prompt, thinking, effort, log)
    except Exception as e:
        error = f"{type(e).__name__}: {S.explain_api_error(e) if hasattr(S, 'explain_api_error') else e}"
        error += "\n" + traceback.format_exc(limit=3)
    return {"model": model, "turns": log, "requests": spy.requests, "error": error,
            "response_models": getattr(spy, "models", []),
            "tool_rounds": sum(1 for t in log if t["tool_calls"])}


def expect_off(res: dict, payload: dict) -> list[str]:
    bad = []
    for i, r in enumerate(res["requests"]):
        if r["thinking"] != payload:
            bad.append(f"req{i} thinking={r['thinking']}")
        if r["output_config"] is not None:
            bad.append(f"req{i} output_config={r['output_config']}")
        if r["streamed"] or r["max_tokens"] != S.ANTHROPIC_MAX_TOKENS:
            bad.append(f"req{i} streamed={r['streamed']} max_tokens={r['max_tokens']}")
    return bad


def expect_ds(res: dict, modes: list[str]) -> list[str]:
    """modes: 'on' | 'off' per turn-phase, in request order is not 1:1 with phases,
    so assert on the set of request shapes actually sent."""
    bad = []
    kinds = set()
    for i, r in enumerate(res["requests"]):
        t = (r["thinking"] or {}).get("type")
        kinds.add(t)
        if t == "disabled" and r["reasoning_effort"] is not None:
            bad.append(f"req{i} disabled but reasoning_effort={r['reasoning_effort']}")
        if t == "enabled" and r["reasoning_effort"] not in ("low", "high", "max"):
            bad.append(f"req{i} enabled with reasoning_effort={r['reasoning_effort']}")
        if t not in ("enabled", "disabled"):
            bad.append(f"req{i} thinking={r['thinking']}")
    want = {"on": {"enabled"}, "off": {"disabled"}, "toggle": {"enabled", "disabled"}}
    for m in modes:
        if kinds != want[m]:
            bad.append(f"request thinking kinds {sorted(k or 'none' for k in kinds)} != {sorted(want[m])}")
    bad += [f"response model {m!r} != 'deepseek-flash'" for m in set(res["response_models"])
            if m != "deepseek-flash"]
    return bad


def check_common(res: dict, min_tool_rounds: int = 2) -> list[str]:
    bad = []
    if res["error"]:
        bad.append(res["error"].splitlines()[0])
    if res["tool_rounds"] < min_tool_rounds:
        bad.append(f"only {res['tool_rounds']} tool rounds (multi-turn replay not exercised)")
    return bad


def main() -> int:
    env_file = S.find_env_file(None)
    if env_file:
        S.load_env_file(env_file)
    import anthropic
    system = S.build_system_prompt(S.gather_host_facts())
    off = (False, "high")
    checks = {
        "sonnet55_off": ("claude-sonnet-5-5", [off, off, off]),
        "sonnet55_toggle": ("claude-sonnet-5-5", [(True, "xhigh"), off, (True, "xhigh")]),
        "haiku55_off": ("claude-haiku-5-5", [off, off, off]),
        "haiku55_on": ("claude-haiku-5-5", [(True, "high"), (True, "high"), (True, "high")]),
        "haiku45_off": ("claude-haiku-4-5-20251001", [off, off, off]),
        "ds_flash_on": ("deepseek-flash", [(True, "high")] * 3),
        "ds_flash_off": ("deepseek-flash", [off, off, off]),
        "ds_flash_toggle": ("deepseek-flash", [(True, "max"), off, (True, "low")]),
        "ds_legacy_toggle": ("deepseek-v4-flash", [(True, "high"), off, (True, "high")]),
    }
    wanted = sys.argv[1:] or list(checks)
    for name in wanted:
        if name not in checks:
            print(f"unknown check {name!r}; choices: {', '.join(checks)}")
            return 2
        key = "DEEPSEEK_API_KEY" if checks[name][0].startswith("deepseek") else "ANTHROPIC_API_KEY"
        if not os.environ.get(key):
            print(f"{key} not set (environment or sys_agent env file); needed by {name}")
            return 2
    results: dict[str, Any] = {
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "anthropic_sdk": anthropic.__version__,
        "checks": {},
    }
    failed = []
    for name in wanted:
        model, phases = checks[name]
        print(f"[{name}] {model} ...", flush=True)
        res = session(model, phases, system)
        bad = check_common(res)
        if name == "sonnet55_off":
            bad += expect_off(res, {"type": "between_tools"})
            if not all(t["status"].startswith("off (between_tools") for t in res["turns"]):
                bad.append("status line not 'off (between_tools...)'")
        elif name == "haiku55_off":
            bad += expect_off(res, {"type": "disabled"})
        elif name == "haiku45_off":
            bad += [f"req{i} thinking={r['thinking']}" for i, r in enumerate(res["requests"]) if r["thinking"] is not None]
        elif name == "sonnet55_toggle":
            kinds = [(r["thinking"] or {}).get("type") for r in res["requests"]]
            if "between_tools" not in kinds or "adaptive" not in kinds:
                bad.append(f"toggle did not exercise both modes: {kinds}")
        elif name == "haiku55_on":
            bad += [f"req{i} not adaptive+streamed" for i, r in enumerate(res["requests"])
                    if (r["thinking"] or {}).get("type") != "adaptive" or not r["streamed"]]
        elif name == "ds_flash_on":
            bad += expect_ds(res, ["on"])
        elif name == "ds_flash_off":
            bad += expect_ds(res, ["off"])
        elif name in ("ds_flash_toggle", "ds_legacy_toggle"):
            bad += expect_ds(res, ["toggle"])
        res["pass"] = not bad
        res["problems"] = bad
        results["checks"][name] = res
        out = sum(t["out"] for t in res["turns"])
        notes = sum(t["thinking_blocks_nonempty"] for t in res["turns"])
        line = (f"[{name}] {'PASS' if not bad else 'FAIL'}  calls={len(res['requests'])} "
                f"tool_rounds={res['tool_rounds']} out_tokens={out} nonempty_thinking_blocks={notes}")
        print(line + ("" if not bad else "\n    " + "\n    ".join(bad)), flush=True)
        if bad:
            failed.append(name)
    state = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    path = state / "sys_agent" / "live_probe_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"RESULT: {'ALL PASS' if not failed else 'FAILURES: ' + str(failed)}  ({path})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
