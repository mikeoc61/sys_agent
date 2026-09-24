# sys_agent

Single-file Python CLI agent: an LLM (Anthropic, OpenAI, or DeepSeek) proposes
shell commands via native tool calling; each runs only after per-command human
approval. `sys_agent.py` + PEP 723 inline deps; runs via `uv run sys_agent.py`
(pip+venv path also documented in `docs/configuration.md`).

## Docs layout
`README.md` is getting-started only, written for a non-technical reader: what
the tool is, the safety model, install, first session, everyday commands.
Reference material lives in `docs/` — `configuration.md` (all `SYS_*` vars,
files, alternative install), `advanced-usage.md` (all meta-commands, consult,
thinking, line editing), `technical-reference.md` (host facts, hardware
identity, SMART, safety layers, audit format, architecture). Keep new detail
out of the README; link to the page that owns it.

## Hard constraints
- **Single file.** Do not split `sys_agent.py` into modules or add deps beyond
  the API SDKs (+ gnureadline on Darwin) without explicit discussion.
- **Approval prompt is the security gate.** The deny-list is a backstop, not a
  boundary.
- **Ctrl-C is non-destructive everywhere** in the REPL (cancels the current
  operation, never ends the session). Quit is explicit: /exit, /quit, Ctrl-D.
- **Import order:** readline must be imported before any SSL-using library
  (openai, anthropic) — macOS segfault-on-exit quirk. Don't reorder the top of
  file.
- **Model changes touch PROVIDER & MODEL CONFIG only** (`DEFAULT_*_MODEL`,
  `CONTEXT_WINDOWS`, `PROVIDER_MODELS`, `PROVIDER_MODEL_PREFIXES`) plus
  provider-specific request quirks in the provider class.

## Targets and backend asymmetries
- Raspberry Pi 5 (Debian) — primary, 24/7 server. Real sessions (`uv run
  --python 3.13`) get **libedit**: uv's standalone CPython links it. The system
  `python3` (3.11) has GNU readline 8.2.
- M3 Mac (macOS 26): real sessions get GNU via gnureadline; the system
  `python3` falls back to libedit.
- EC2 Ubuntu (Xen and Nitro).
- Changes touching readline, terminal, signals, or process management must be
  tested on (or reasoned about for) both GNU readline and libedit. Pi-only bugs
  are real (v1.4.1 history fix).

## Architecture notes
- Providers share Turn / Usage / CmdResult / ChatTurn shapes. Anthropic batches
  tool_results per assistant turn; OpenAI/DeepSeek send one `role: tool` per
  call. Don't unify message formats without thinking this through.
- **Stop-reason contract:** every `chat()` classifies `stop_reason` /
  `finish_reason` into `ChatTurn.stop` (`""` | `refusal` | `truncated`) with a
  user-facing `stop_detail`. Branch on `stop_reason`, never on `stop_details`
  (populated only on refusal; category/explanation may be null). The REPL
  discards a stopped turn's tool calls BEFORE the deny check and approval
  prompt — a `tool_use` cut mid-input parses as a valid shorter command, so
  the stop reason is the only signal. `tool_calls` stays populated so the loop
  can tell a cut command from a cut text answer. No server-side fallbacks
  (`fallbacks` param) without discussion: they change which model runs.
- `DeepSeekProvider` subclasses `OpenAIProvider` (OpenAI-compatible endpoint).
  **Replay contract:** with `tools` present, `reasoning_content` must be on
  every replayed assistant message or the next call 400s.
- **Default-on thinking:** Sonnet 5, Opus 5, and DeepSeek think when no field is
  sent. The `/thinking off` path must send an explicit disabled. Verify any new
  model's default before assuming off means off.
- Anthropic caching: `input_tokens` excludes cached tokens. True context =
  input + cache_read + cache_write. Display logic depends on this.
- GPT-5.6 on Chat Completions requires `reasoning_effort: "none"` (via
  `extra_body`) to use function tools.

## Startup facts (host priors)
Three-filter rule — reject an addition if it is (1) already collected,
(2) privacy/security-excluded, or (3) volatile/live (belongs in a command, not a
startup prior). E.g. battery SoH passes; SoC fails.

## Prompt rules
- Negative capability claims require a verified probe; the agent must not
  assert a tool is absent without checking.
- Defer new rules until a failure recurs; one occurrence is not enough.

## Verification
- Baseline: `python3 -m py_compile sys_agent.py && python3 test_consult_render.py`
  (offline, no keys; must print `RESULT: ALL PASS`).
- Hardware harness: `test_hardware.py` — offline and keyless, but needs a pty
  and a real process table, so it is per-host and NOT part of the baseline.
  Run it on the Pi and the Mac before tagging, under BOTH interpreters:
  `uv run --python 3.13 test_hardware.py` (the backend real sessions use) and
  `python3 test_hardware.py` (the other one). Covers what the
  baseline structurally cannot: signal delivery and process-group teardown, the
  approval prompt under real readline, env-file resolution at real startup. It
  reports the live readline backend, so a run records which of the two paths
  was exercised.
- **NO-SPAWN invariant in `test_hardware.py`:** its REPL checks stub
  `execute()`, the single point where an approved command becomes a subprocess.
  Never restore the real one there. A mutation check on a safety gate must
  neutralize the payload in the same step that removes the gate — removing the
  gate is exactly what makes the payload live. (Learned the hard way: a
  mutation run once put a live `rm -rf /` through the execution path on the Pi,
  stopped only by coreutils' `--preserve-root`.)
- Extend `test_consult_render.py` for wire-format changes and `test_hardware.py`
  for anything touching signals, process management, readline, or startup
  ordering; prefer driving real code with fake clients/providers over isolated
  unit tests. Mutation-check new guards (revert the fix, confirm the test
  fails), and make the mutation reproduce the ORIGINAL bug — a mutation that
  merely reshapes the code can still pass and prove nothing.
- Validate claims against raw API/command output, not model self-reports.
- Before asserting a provider API behavior, probe it (curl) or cite current
  docs. Provider defaults change without notice.
- Compiling is not proof; a behavioral test is.

## Git and release
- Semver `vX.Y.Z`, annotated tags. Minor: new model support, new facts blocks.
  Patch: probe-list gaps, prompt-only, docs-only.
- README and the `docs/` page that owns the affected material updated in the
  same commit as the code; system prompt alongside prompt-rule changes. Single
  commits unless a split helps bisect.
- Commit subject: terse, imperative. Tag message: substantive (what and why,
  non-obvious reasoning such as replay contracts).
- Commit freely; **tag only after Pi validation** — both harnesses green on the
  Pi (display-only changes may skip hardware re-test). Never force-update a published tag. Push tags explicitly:
  `git push origin <tag>`. Check `git describe --tags` before choosing a number.
- Do not tag or push without explicit instruction.

## Style
- Type hints throughout. Information-dense comments that record non-obvious
  reasoning; no filler.
- Shell commands in docs: no inline or trailing comments.
