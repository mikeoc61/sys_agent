"""Offline round-trip verification for the /consult render layer + fork-from-
question trim. No network, no API keys — exercises render/normalize/trim only,
never .chat(). Run: python3 -m py_compile sys_agent.py && python3 test_consult_render.py
"""
import json, sys
import sys_agent as S

def _mk(cls, name, model):
    p = object.__new__(cls)            # bypass __init__ (builds an SDK client)
    p.name, p.model = name, model
    return p

OAI = _mk(S.OpenAIProvider, "openai", "gpt-5.4-mini")
DS  = _mk(S.DeepSeekProvider, "deepseek", "deepseek-flash")
ANT = _mk(S.AnthropicProvider, "anthropic", "claude-haiku-4-5-20251001")
PROVIDERS = {"openai": OAI, "deepseek": DS, "anthropic": ANT}
SYS = "SYSTEM PROMPT TEXT"

TID1, TID2 = "toolu_01ABC", "call_xyz789"
events = [
    S.CanonicalEvent("user", text="why is bitcoind restarting?"),
    S.CanonicalEvent("assistant", text="Let me check the service.",
        tool_calls=[S.ToolCall(TID1, "run_command",
            {"command": "systemctl status bitcoind", "explanation": "check unit"})]),
    S.CanonicalEvent("tool_results", results=[(TID1, json.dumps({"returncode":0,"stdout":"active"}))]),
    S.CanonicalEvent("assistant", text="",
        tool_calls=[S.ToolCall(TID2, "run_command",
            {"command": "journalctl -u bitcoind -n 50", "explanation": "recent logs"})]),
    S.CanonicalEvent("tool_results", results=[(TID2, json.dumps({"returncode":0,"stdout":"OOM"}))]),
    S.CanonicalEvent("assistant", text="It was OOM-killed. Raise dbcache or add swap."),
]

def canon_eq(a, b):
    if len(a) != len(b): return False
    for x, y in zip(a, b):
        if x.kind != y.kind or x.text != y.text: return False
        if [(t.id,t.name,t.arguments) for t in x.tool_calls] != \
           [(t.id,t.name,t.arguments) for t in y.tool_calls]: return False
        if x.results != y.results: return False
    return True

fails = []

# 1) render→normalize identity per provider
for name, p in PROVIDERS.items():
    back = S.normalize(name, S.render(p, events, SYS))
    ok = canon_eq(events, back)
    print(f"[{name}] render→normalize identity: {'OK' if ok else 'FAIL'}")
    if not ok: fails.append(name)

# 2) cross-provider consult path: active=anthropic → render to each target → normalize stable
active_events = S.normalize("anthropic", S.render(ANT, events, SYS))
for name, p in PROVIDERS.items():
    if name == "anthropic": continue
    ok = canon_eq(active_events, S.normalize(name, S.render(p, active_events, SYS)))
    print(f"[anthropic→{name}] cross-provider consult render: {'OK' if ok else 'FAIL'}")
    if not ok: fails.append(f"x-{name}")

# 3) wire invariants the live APIs enforce
oai_wire = S.render(OAI, events, SYS)
assert oai_wire[0] == {"role":"system","content":SYS}, "OpenAI system placement"
for m in (m for m in oai_wire if m.get("role")=="assistant"):
    for tc in m.get("tool_calls", []):
        assert tc["type"]=="function" and isinstance(tc["function"]["arguments"], str)
        json.loads(tc["function"]["arguments"])
oai_from_ant = S.render(OAI, active_events, SYS)
tids = [m["tool_call_id"] for m in oai_from_ant if m.get("role")=="tool"]
assert TID1 in tids and TID2 in tids, f"tool id verbatim: {tids}"
print("[openai] wire invariants (system/tc-shape/id-preservation): OK")
ds_asst = [m for m in S.render(DS, active_events, SYS) if m.get("role")=="assistant"]
assert ds_asst and all("reasoning_content" in m for m in ds_asst), "DeepSeek reasoning_content presence"
print(f"[deepseek] reasoning_content on all {len(ds_asst)} assistant msgs: OK")
aw = S.render(ANT, active_events, SYS)
assert all(isinstance(m["content"], list) for m in aw if m["role"]=="assistant")
assert not any(m.get("role")=="system" for m in aw)
batched = [m for m in aw if m["role"]=="user" and isinstance(m["content"],list)
           and any(b.get("type")=="tool_result" for b in m["content"])]
assert len(batched)==2, f"Anthropic batched tool_results: {len(batched)}"
print("[anthropic] block-list / system-placement / tool_result batching: OK")

# 4) empty-assistant guard
g = S.render(ANT, [S.CanonicalEvent("user",text="hi"), S.CanonicalEvent("assistant")], SYS)
assert g[-1]["content"] == [{"type":"text","text":" "}], f"empty guard: {g[-1]}"
print("[anthropic] empty-assistant content guard: OK")

# 5) fork-from-question trim — the bug fix. Full transcript ends on an assistant
#    final answer (Haiku returned empty). After trim, each provider's rendered
#    input must end on the USER question, with the active answer removed.
last_user = max(i for i, ev in enumerate(events) if ev.kind == "user")
trimmed = events[:last_user + 1]
assert trimmed[-1].kind == "user"
for name, p in PROVIDERS.items():
    wire = S.render(p, trimmed, SYS)
    if wire[-1]["role"] != "user":
        print(f"[trim:{name}] FAIL last role {wire[-1]['role']}"); fails.append(f"trim-{name}"); continue
    if "OOM-killed" in json.dumps(wire) or "Raise dbcache" in json.dumps(wire):
        print(f"[trim:{name}] FAIL active answer leaked"); fails.append(f"trim-{name}"); continue
    print(f"[trim:{name}] ends on user question, active answer removed: OK")

# reference resolves to active provider's first move on the trimmed tail
ref = []
for ev in events[last_user+1:]:
    if ev.kind == "assistant" and ev.tool_calls:
        ref = [tc.arguments.get("command","").strip() for tc in ev.tool_calls if tc.name=="run_command"]
        break
if ref[:1] == ["systemctl status bitcoind"]:
    print("[trim] reference = active provider's first command: OK")
else:
    print(f"[trim] reference FAIL: {ref}"); fails.append("trim-ref")

# 6) abort→consult: an aborted question is rolled out of `messages`; the
#    capture must let /consult re-pose IT, not the prior surviving turn.
m = DS.initial_messages(SYS)
m.append({"role":"user","content":"yes, please upgrade them"})
m.append(DS.render_assistant("", [S.ToolCall("c_up","run_command",
    {"command":"brew upgrade","explanation":"upgrade"})]))
DS.append_tool_results(m,[("c_up",'{"stdout":"ok"}')])
m.append(DS.render_assistant("Done.", []))
ts = len(m)
m.append({"role":"user","content":"how's my security posture?"})
m.append(DS.render_assistant("checking", [S.ToolCall("c_sec","run_command",
    {"command":"csrutil status","explanation":"SIP"})]))
captured = S.normalize(DS.name, m[ts:])          # capture before rollback
del m[ts:]                                        # rollback
# buggy path would pick the prior surviving question:
ev_old = S.normalize(DS.name, m)
old_q = ev_old[max(i for i,e in enumerate(ev_old) if e.kind=="user")].text
# fixed path uses the captured aborted question:
new_q = next(e for e in captured if e.kind=="user").text
new_ref = S._first_move_commands(captured)
ok = (old_q == "yes, please upgrade them"
      and new_q == "how's my security posture?"
      and new_ref and new_ref[0][0] == "csrutil status")
print(f"[abort→consult] re-poses aborted question (not prior turn): {'OK' if ok else 'FAIL'}")
if not ok: fails.append("abort-consult")

# 7) --fresh drops prior history; default retains it (both end on the question)
hist = [
    S.CanonicalEvent("user", text="brew up to date?"),
    S.CanonicalEvent("assistant", text="", tool_calls=[S.ToolCall("h1","run_command",{"command":"brew outdated","explanation":"x"})]),
    S.CanonicalEvent("tool_results", results=[("h1",'{"stdout":"ok"}')]),
    S.CanonicalEvent("assistant", text="done"),
    S.CanonicalEvent("user", text="how's my security posture?"),
    S.CanonicalEvent("assistant", text="ans"),
]
lu = max(i for i,e in enumerate(hist) if e.kind=="user")
q = S.CanonicalEvent("user", text=hist[lu].text)
default_trim = hist[:lu] + [q]
fresh_trim = [q]
ok_fresh = True
for name,p in PROVIDERS.items():
    wd, wf = json.dumps(S.render(p, default_trim, SYS)), json.dumps(S.render(p, fresh_trim, SYS))
    if not ("brew outdated" in wd and "brew outdated" not in wf): ok_fresh=False
    if "security posture" not in wf: ok_fresh=False
    if S.render(p, fresh_trim, SYS)[-1]["role"] != "user": ok_fresh=False
    if len(wf) >= len(wd): ok_fresh=False
parse_ok = all(
    any(a in ("--fresh","fresh","-f") for a in s.split()[1:]) == exp
    for s,exp in [("/consult",False),("/consult --fresh",True),("/consult fresh",True),("/consult -f",True),("/consult --bogus",False)]
)
print(f"[fresh] --fresh trims history, default retains it, flag parses: {'OK' if (ok_fresh and parse_ok) else 'FAIL'}")
if not (ok_fresh and parse_ok): fails.append("fresh")

# 8) DeepSeek thinking wire format. Server default is thinking ON, so the off
#    path must send an explicit disabled; on path sends the mapped grade.
from types import SimpleNamespace as _NS
class _Msg:
    content, tool_calls, reasoning_content = "ok", None, ""
    def model_dump(self, exclude_none=True): return {"role": "assistant", "content": "ok"}
class _Client:
    def __init__(self): self.kw, self.chat, self.completions = None, self, self
    def create(self, **kw):
        self.kw = kw
        return _NS(choices=[_NS(message=_Msg())], usage=_NS(prompt_tokens=1, completion_tokens=1))
DS.client = _Client()
DS.chat([], SYS, thinking=False)
ok_off = DS.client.kw.get("extra_body") == {"thinking": {"type": "disabled"}}
exp = {"low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}
ok_on = True
for lvl, grade in exp.items():
    DS.chat([], SYS, thinking=True, effort=lvl)
    if DS.client.kw.get("extra_body") != {"thinking": {"type": "enabled"}, "reasoning_effort": grade}:
        ok_on = False
ok_lv = S._effort_usage_levels(DS) == ("low", "high", "max")
ok_note = (S.effective_effort(DS, "low") == ("low", "")
           and S.effective_effort(DS, "medium")[1] != ""
           and S.effective_effort(DS, "max") == ("max", ""))
ok_ds = ok_off and ok_on and ok_lv and ok_note
print(f"[deepseek] thinking off->disabled, effort low|high|max mapping: {'OK' if ok_ds else 'FAIL'}")
if not ok_ds: fails.append("ds-thinking")

# 9) model-table consistency: every listed model has a context window, and
#    every built-in provider default is a listed model (skipped per provider
#    when SYS_<PROVIDER>_MODEL overrides it in the environment).
import os
missing = [m for ms in S.PROVIDER_MODELS.values() for m in ms if m not in S.CONTEXT_WINDOWS]
unlisted = [f"{p}:{m}" for p, m in S.DEFAULT_MODELS.items()
            if not os.environ.get(f"SYS_{p.upper()}_MODEL") and m not in S.PROVIDER_MODELS[p]]
ok_tab = not missing and not unlisted
print(f"[models] PROVIDER_MODELS/CONTEXT_WINDOWS/DEFAULT_MODELS consistency: {'OK' if ok_tab else f'FAIL {missing} {unlisted}'}")
if not ok_tab: fails.append("model-tables")

# 10) tool-argument validation. The model controls this payload and the wire
#     schema is not enforced: json.loads of a bare array/scalar succeeds, and a
#     field can hold a non-string. The tool loop sits outside the turn's
#     try/except, so an uncaught AttributeError here ends the session.
arg_cases = [
    ({"command": "ls -l", "explanation": "list"}, ("ls -l", "list", None)),
    ({"command": "  ls  ", "explanation": None},  ("ls", "", None)),
    ({"command": "ls"},                           ("ls", "", None)),
]
ok_args = all(S.parse_command_args(a) == exp for a, exp in arg_cases)
# Malformed shapes must come back as errors, not exceptions.
bad = [[1, 2], "ls", 5, None, {"command": ["ls", "-l"]}, {"command": 5},
       {"command": ""}, {"command": "   "}, {}]
for b in bad:
    try:
        cmd, why, e = S.parse_command_args(b)
    except Exception as exc:                        # noqa: BLE001
        ok_args = False
        print(f"   raised on {b!r}: {type(exc).__name__}")
        continue
    if not e or cmd:
        ok_args = False
        print(f"   accepted malformed {b!r} -> {(cmd, why, e)!r}")
# A non-string explanation is demoted, not rejected: the reason line is
# cosmetic and must not fail an otherwise valid command.
ok_args = ok_args and S.parse_command_args(
    {"command": "ls", "explanation": 7}) == ("ls", "", None)
print(f"[toolargs] malformed tool arguments return errors, never raise: {'OK' if ok_args else 'FAIL'}")
if not ok_args: fails.append("tool-args")

# 11) deny check covers the EDITED command. The check in the tool loop runs
#     against the model's proposal; an edit replaces that string, so the
#     approved-and-edited form is what must be re-checked before spawning.
#     Mutation check: drop the re-check in the loop and this still passes
#     (it asserts is_denied, not the call site) -- the call site is covered by
#     the audit/flow shape, so keep both in mind when editing run_repl.
ok_deny = (S.is_denied("rm -rf /") is not None
           and S.is_denied("ls -l") is None
           and S.is_denied("sudo rm -fr /") is not None)
import inspect
_loop_src = inspect.getsource(S.run_repl)
# The edited command must be passed through is_denied before execute().
_edit_idx = _loop_src.find("to_run = edited or cmd")
_exec_idx = _loop_src.find("execute(to_run)")
ok_recheck = (_edit_idx != -1 and _exec_idx != -1
              and "is_denied(to_run)" in _loop_src[_edit_idx:_exec_idx])
ok_deny = ok_deny and ok_recheck
print(f"[deny-edit] edited command is re-checked before execute: {'OK' if ok_deny else 'FAIL'}")
if not ok_deny: fails.append("deny-edit")

# 12) process-group cleanup. The shell almost always dies on the first SIGTERM,
#     so its exit says nothing about whether the group is clear. A descendant
#     that ignores SIGTERM must still be SIGKILLed (this returned on the shell's
#     exit before, so SIGKILL was never sent), while a well-behaved group must
#     not pay the grace period.
import subprocess, time as _time, os as _os, tempfile

def _alive(pid):
    try:
        _os.kill(pid, 0); return True
    except (ProcessLookupError, OSError):
        return False

def _run_group(body):
    """Spawn `body` under a shell in its own session; return (proc, child_pid)."""
    pf = tempfile.NamedTemporaryFile("r", suffix=".pid", delete=False)
    pf.close()
    proc = subprocess.Popen(body.format(pid=pf.name), shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    for _ in range(50):                      # wait for the child to register
        _time.sleep(0.1)
        try:
            txt = open(pf.name).read().strip()
            if txt:
                return proc, int(txt)
        except (OSError, ValueError):
            pass
    return proc, -1

_IGNORER = ('python3 -c "import signal,os,time,sys;'
            'signal.signal(signal.SIGTERM, signal.SIG_IGN);'
            "open('{pid}','w').write(str(os.getpid()));"
            'time.sleep(60)" & sleep 60')
_WELLBEHAVED = ('python3 -c "import os,time;'
                "open('{pid}','w').write(str(os.getpid()));"
                'time.sleep(60)" & sleep 60')

proc, gp = _run_group(_IGNORER)
ok_kill = gp > 0 and _alive(gp)
S._terminate_group(proc)
_time.sleep(0.5)
ok_kill = ok_kill and not _alive(gp)
if not ok_kill and gp > 0 and _alive(gp):
    _os.kill(gp, 9)

proc2, gp2 = _run_group(_WELLBEHAVED)
_t0 = _time.monotonic()
S._terminate_group(proc2)
_fast = _time.monotonic() - _t0
_time.sleep(0.3)
# Well-behaved group: dies on SIGTERM and returns well inside the grace window.
ok_fast = gp2 > 0 and not _alive(gp2) and _fast < S._GROUP_TERM_GRACE
if gp2 > 0 and _alive(gp2):
    _os.kill(gp2, 9)

# An already-reaped process must not raise or hang.
p3 = subprocess.Popen("true", shell=True, stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE, text=True, start_new_session=True)
p3.wait()
try:
    S._terminate_group(p3)
    ok_reaped = True
except Exception:                                   # noqa: BLE001
    ok_reaped = False

ok_pg = ok_kill and ok_fast and ok_reaped
print(f"[killgroup] SIGTERM-ignoring descendant killed ({'OK' if ok_kill else 'FAIL'}), "
      f"clean group fast-path {_fast:.2f}s ({'OK' if ok_fast else 'FAIL'}), "
      f"reaped proc safe ({'OK' if ok_reaped else 'FAIL'})")
if not ok_pg: fails.append("kill-group")

# 13) config resolution runs AFTER the env file is loaded. Every SYS_* setting
#     used to be captured at import, before main() read the file, so a
#     documented .env value was silently ignored. Shell vars must still win,
#     and a bad value must be reported rather than abort startup.
_saved = {k: os.environ.get(k) for k in
          ("SYS_COMMAND_TIMEOUT", "SYS_THINKING", "SYS_TOP_PROCESSES",
           "SYS_THINKING_EFFORT")}
_orig = (S.COMMAND_TIMEOUT, S.ANTHROPIC_THINKING_DEFAULT,
         S.RUNTIME_TOP_PROCESSES, S.ANTHROPIC_THINKING_EFFORT_DEFAULT)
try:
    # Simulate load_env_file having merged these in, then re-resolve.
    os.environ["SYS_COMMAND_TIMEOUT"] = "7"
    os.environ["SYS_THINKING"] = "on"
    probs = S.init_config()
    ok_cfg = (S.COMMAND_TIMEOUT == 7 and S.ANTHROPIC_THINKING_DEFAULT is True
              and not probs)
    # Bad values: reported, ignored, never raised.
    os.environ["SYS_COMMAND_TIMEOUT"] = "abc"
    os.environ["SYS_TOP_PROCESSES"] = "-4"
    os.environ["SYS_THINKING_EFFORT"] = "turbo"
    probs = S.init_config()
    ok_bad = (len(probs) == 3 and S.COMMAND_TIMEOUT == 7
              and S.RUNTIME_TOP_PROCESSES == _orig[2]
              and S.ANTHROPIC_THINKING_EFFORT_DEFAULT in S._VALID_EFFORTS)
    # Derived table tracks the model globals.
    os.environ["SYS_ANTHROPIC_MODEL"] = "claude-sonnet-5"
    S.init_config()
    ok_derived = S.DEFAULT_MODELS["anthropic"] == "claude-sonnet-5"
finally:
    os.environ.pop("SYS_ANTHROPIC_MODEL", None)
    for k, v in _saved.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
    S.init_config()
ok_conf = ok_cfg and ok_bad and ok_derived
print(f"[config] env file values applied, bad values reported not fatal: {'OK' if ok_conf else 'FAIL'}")
if not ok_conf: fails.append("config-init")

# 14) gpt-6-astra's tools 400 tells the user to "set reasoning_effort to
#     'none'", which that model rejects; the raw text used to reach the REPL
#     untranslated. The unlisted-model warning must drop its context-% hedge
#     when the window is recorded.
class _FakeAPIError(Exception):
    # Stand-in for the SDK's BadRequestError (the baseline runs without the
    # SDKs installed): same status_code attribute and str() shape.
    def __init__(self, msg: str, status_code: int) -> None:
        super().__init__(msg); self.status_code = status_code
# Verbatim str(e) from a live M3 session, 2026-09-23.
_e = _FakeAPIError(
    "Error code: 400 - {'error': {'message': \"Function tools with "
    "reasoning_effort are not supported for gpt-6-astra in "
    "/v1/chat/completions. To use function tools, use /v1/responses or set "
    "reasoning_effort to 'none'.\", 'type': 'invalid_request_error', "
    "'param': 'reasoning_effort', 'code': None}}", 400)
_x = S.explain_api_error(_e)
ok_x = ("gpt-6-astra cannot call tools" in _x and "/models" in _x
        and "set reasoning_effort to 'none'" not in _x)
# Unrelated 400s still fall through to the raw message.
_e2 = _FakeAPIError("Error code: 400 - other", 400)
ok_x = ok_x and S.explain_api_error(_e2) == "Error code: 400 - other"
_w_known = S.unlisted_model_warning("openai", "gpt-6-astra")
_w_unknown = S.unlisted_model_warning("openai", "gpt-9-nova")
ok_w = ("context-%" not in _w_known and "context-% may be unavailable"
        in _w_unknown)
print(f"[apierr] astra tools-400 translated: {'OK' if ok_x else 'FAIL: ' + _x}")
print(f"[apierr] unlisted-model warning hedges only on unknown window: "
      f"{'OK' if ok_w else 'FAIL'}")
if not ok_x: fails.append("apierr-astra")
if not ok_w: fails.append("unlisted-warning")

# 15) reasoning_effort="none" routing through the real OpenAIProvider.chat().
#     gpt-6-luna needs it or every tools turn 400s; gpt-6-astra must NOT get
#     it (a bare "gpt-6" prefix would send it, turning Astra's translated
#     tools 400 into an untranslated unsupported_value 400). DeepSeek inherits
#     chat() plumbing and must stay untouched.
class _FakeCompletions:
    def __init__(self) -> None: self.kwargs: dict = {}
    def create(self, **kw):
        self.kwargs = kw
        msg = _NS(content="ok", tool_calls=None,
                  model_dump=lambda **_: {"role": "assistant", "content": "ok"})
        return _NS(choices=[_NS(message=msg)],
                   usage=_NS(prompt_tokens=1, completion_tokens=1))
def _sent_effort(model: str) -> object:
    p = _mk(S.OpenAIProvider, "openai", model)
    fc = _FakeCompletions()
    p.client = _NS(chat=_NS(completions=fc))
    p.chat([{"role": "user", "content": "hi"}], SYS)
    return fc.kwargs.get("extra_body", {}).get("reasoning_effort")
#     Models dropped from the list (gpt-4o-mini, GPT-5.6) keep their window
#     and, for GPT-5.6, the forced "none", so existing pins keep working.
_want = {"gpt-6-luna": "none", "gpt-6-sol": "none", "gpt-5.6-luna": "none",
         "gpt-5.6-terra": "none", "gpt-6-astra": None, "gpt-5.4-mini": None,
         "gpt-4.1-nano": None}
_eff = {m: _sent_effort(m) for m in _want}
_listed = S.PROVIDER_MODELS["openai"]
ok_eff = (_eff == _want
          and _listed == ("gpt-4.1-nano", "gpt-5.4-mini", "gpt-6-luna",
                          "gpt-6-sol")
          and all(m in S.CONTEXT_WINDOWS for m in
                  ("gpt-6-luna", "gpt-6-sol", "gpt-4o-mini", "gpt-5.6-luna",
                   "gpt-5.6-terra")))
print(f"[openai] effort=none routing (luna/sol yes, astra no), list: "
      f"{'OK' if ok_eff else f'FAIL {_eff}'}")
if not ok_eff: fails.append("effort-none-routing")

# 16) Anthropic thinking routing through the real AnthropicProvider.chat().
#     Opus 5.5 / Fable 5.x 400 on thinking=disabled (probe 2026-09-24), so the
#     thinking-off path must never send it to them; it must instead give them
#     the thinking-sized cap on the streaming path (a 4096 cap can truncate a
#     turn that thinks anyway). Sonnet 5 / Opus 5 still need the explicit
#     disabled to honor /thinking off; Haiku sends nothing. Status must never
#     read "off" for a model that cannot stop thinking.
class _FakeMessages:
    def __init__(self) -> None:
        self.kwargs: dict = {}; self.streamed = False
    def _resp(self):
        blk = _NS(type="text", text="ok",
                  model_dump=lambda **_: {"type": "text", "text": "ok"})
        return _NS(content=[blk], usage=_NS(
            input_tokens=1, output_tokens=1, cache_read_input_tokens=0,
            cache_creation_input_tokens=0))
    def create(self, **kw):
        self.kwargs = kw; return self._resp()
    def stream(self, **kw):
        self.kwargs = kw; self.streamed = True; outer = self
        class _S:
            def __enter__(self): return _NS(get_final_message=outer._resp)
            def __exit__(self, *a): return False
        return _S()
def _ant_req(model: str, thinking: bool) -> tuple[object, int, bool]:
    p = _mk(S.AnthropicProvider, "anthropic", model)
    fm = _FakeMessages()
    p.client = _NS(messages=fm)
    p.chat([{"role": "user", "content": "hi"}], SYS, thinking=thinking,
           effort="high")
    th = (fm.kwargs.get("extra_body", {}).get("thinking")
          or fm.kwargs.get("thinking"))
    return th, fm.kwargs["max_tokens"], fm.streamed
_BIG, _SMALL = S.ANTHROPIC_THINKING_MAX_TOKENS, S.ANTHROPIC_MAX_TOKENS
_ant_want = {
    ("claude-opus-5-5", False): (None, _BIG, True),
    ("claude-fable-5-1", False): (None, _BIG, True),
    ("claude-opus-5-5", True): ({"type": "adaptive"}, _BIG, True),
    ("claude-opus-5", False): ({"type": "disabled"}, _SMALL, False),
    ("claude-sonnet-5", False): ({"type": "disabled"}, _SMALL, False),
    ("claude-haiku-4-5-20251001", False): (None, _SMALL, False),
}
_ant_got = {k: _ant_req(*k) for k in _ant_want}
_prov = lambda m: _mk(S.AnthropicProvider, "anthropic", m)
ok_ant = (_ant_got == _ant_want
          and S.thinking_status(_prov("claude-opus-5-5"), False)
              == "on (always on for claude-opus-5-5)"
          and S.thinking_active(_prov("claude-opus-5-5"), False)
          and S.thinking_status(_prov("claude-haiku-4-5-20251001"), False) == "off"
          and not S.thinking_active(_prov("claude-haiku-4-5-20251001"), False)
          and not (S._THINKING_ALWAYS_ON_MODELS & S._ADAPTIVE_DEFAULT_ON_MODELS)
          and S.PROVIDER_MODELS["anthropic"] == (
              "claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-5-5")
          and all(m in S.CONTEXT_WINDOWS for m in
                  ("claude-opus-5-5", "claude-fable-5-1", "claude-opus-5",
                   "claude-opus-4-8")))
_ant_bad = {k: v for k, v in _ant_got.items() if v != _ant_want[k]}
print(f"[anthropic] thinking routing (always-on / default-on / off), list: "
      f"{'OK' if ok_ant else f'FAIL {_ant_bad}'}")
if not ok_ant: fails.append("anthropic-thinking-routing")

# 17) DeepSeek: deepseek-v4-pro is listed again (DeepSeek reversed its
#     2026-09-14 routing to Flash). Both listed models think by default, so
#     /thinking off must send an explicit disabled to each; /thinking on sends
#     enabled + the mapped effort grade. Drives the real DeepSeekProvider.chat().
def _ds_req(model: str, thinking: bool) -> dict:
    p = _mk(S.DeepSeekProvider, "deepseek", model)
    fc = _FakeCompletions()
    p.client = _NS(chat=_NS(completions=fc))
    p.chat([{"role": "user", "content": "hi"}], SYS, thinking=thinking,
           effort="xhigh")
    return fc.kwargs.get("extra_body", {})
_ds_want = {
    ("deepseek-flash", False): {"thinking": {"type": "disabled"}},
    ("deepseek-v4-pro", False): {"thinking": {"type": "disabled"}},
    ("deepseek-v4-pro", True): {"thinking": {"type": "enabled"},
                                "reasoning_effort": "max"},
}
_ds_got = {k: _ds_req(*k) for k in _ds_want}
ok_ds = (_ds_got == _ds_want
         and S.PROVIDER_MODELS["deepseek"] == ("deepseek-flash",
                                               "deepseek-v4-pro")
         and S.DEFAULT_DEEPSEEK_MODEL == "deepseek-flash"
         and all(m in S.CONTEXT_WINDOWS for m in
                 ("deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash")))
print(f"[deepseek] v4-pro listed, thinking off/on routing: "
      f"{'OK' if ok_ds else f'FAIL {_ds_got}'}")
if not ok_ds: fails.append("deepseek-v4-pro")

print()
print("RESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
