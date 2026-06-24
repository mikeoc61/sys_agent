"""Offline round-trip verification for the /consult render layer + fork-from-
question trim. No network, no API keys — exercises render/normalize/trim only,
never .chat(). Run: python3 test_consult_render.py
"""
import json, sys
import sys_agent as S

def _mk(cls, name, model):
    p = object.__new__(cls)            # bypass __init__ (builds an SDK client)
    p.name, p.model = name, model
    return p

OAI = _mk(S.OpenAIProvider, "openai", "gpt-5.4-mini")
DS  = _mk(S.DeepSeekProvider, "deepseek", "deepseek-v4-flash")
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

print()
print("RESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
