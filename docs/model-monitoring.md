# Model monitoring

[Getting started](../README.md) · [Configuration](configuration.md) ·
[Advanced usage](advanced-usage.md) · [Technical reference](technical-reference.md)

`tools/model_watch.py` is a separate, standard-library-only checker for a
monitoring agent. It reads sys_agent's curated defaults, model lists and context
sizes, then asks each configured provider for its current model inventory. It
does not call a chat model, alter sys_agent's choices, or spend generation
tokens. Its only write is a private snapshot at
`~/.local/state/sys_agent/model_watch.json` (or under `$XDG_STATE_HOME`).

Run it on the machine whose API keys and sys_agent checkout you want to check:

```bash
python3 tools/model_watch.py
```

The output is one JSON object. `status: baseline` means the first successful
check for that provider; later checks report `new_models`,
`disappeared_models`, and `metadata_changes` relative to the saved snapshot.
`listed_not_visible` and `context_review` compare the current API inventory
with sys_agent's curated list. `needs_review` describes any current discrepancy;
`new_signal` is true only when something newly changes and is the appropriate
trigger for a notification. `coverage_incomplete` means at least one provider
could not be checked; this is expected for providers whose keys you have not
configured. A failed request leaves that provider's prior inventory intact,
so the next successful run can still detect the change. Provider health is
stored alongside the inventory: `consecutive_failures` resets after a successful
check, and `coverage_needs_repair` becomes true after two consecutive failures.

The checker uses the same API-key lookup as sys_agent: shell variables first,
then `SYS_ENV_FILE` or its normal `.env` search. Run from the checkout if you
keep `.env` there; a monitoring agent with another working directory can set
`SYS_ENV_FILE` to its absolute path. API keys are not written to the snapshot
or output. The `--state PATH` option selects another snapshot and `--no-state`
avoids writing one.

## OpenClaw on the Pi

After this checkout is on the Pi, give the OpenClaw agent read and execution
access to the checkout and schedule a weekly *agent-turn* automation. OpenClaw
supports scheduled agent turns and delivery to a configured chat channel;
choose the existing agent and delivery route on that Pi. The monitor's task
should say:

> Run `python3 /absolute/path/to/sys_agent/tools/model_watch.py`. Read the JSON
> report. If `new_signal` and `coverage_needs_repair` are false, stay quiet
> unless `checked_providers` is zero. For a new model,
> changed metadata, or a missing curated model,
> inspect official provider model documentation, release notes, pricing, and
> the relevant sys_agent request path. Explain what changed, whether it affects
> the default or an existing model, what evidence supports an update, and what
> tool-call or thinking-mode test is still needed. Notify me only when a
> concrete update is warranted or when coverage needs repair. Do not edit
> sys_agent or switch its default model automatically.

Provider inventory is an early signal, not a support verdict. OpenAI's list
provides little capability metadata; model IDs there can include models meant
for other tasks. An ID absent from an account's list may reflect access rather
than retirement. DeepSeek's unversioned `deepseek-flash` alias may change its
underlying revision without changing its ID or metadata, so the monitoring
agent should also review the [DeepSeek changelog](https://api-docs.deepseek.com/updates/).
The provider endpoints used by the checker are documented by
[OpenAI](https://platform.openai.com/docs/api-reference/models/list),
[Anthropic](https://platform.claude.com/docs/en/api/models/list), and
[DeepSeek](https://api-docs.deepseek.com/api/list-models/).

## Live contract probe

`tools/live_probe.py` closes the gap the inventory check leaves: it confirms
that sys_agent's own request shapes are accepted by a model before that model
is listed or made the default. It drives the real `AnthropicProvider.chat()` (or
`DeepSeekProvider.chat()`) tool loop through three prompts per check and records each request's thinking
payload, effort, token cap and streaming path, plus stop reasons, thinking
blocks and token usage per call.

```bash
uv run tools/live_probe.py
uv run tools/live_probe.py sonnet55_off sonnet55_toggle
```

It spends real tokens (about $0.13 for the five Anthropic checks, mostly Sonnet 5.5; the four DeepSeek checks are cents) but
never executes a command: every `run_command` call is answered with canned
output, so nothing runs on the host and the approval gate is not involved. It
reads the API key the same way sys_agent does. Results go to
`~/.local/state/sys_agent/live_probe_results.json` (or under
`$XDG_STATE_HOME`); the exit status is 0 only if every check passes.

The checks cover the thinking-off forms that differ per model: `between_tools`
on Sonnet 5.5, `disabled` on Haiku 5.5, a Sonnet 5.5 on→off→on toggle over one
history (thinking-block replay across mode switches), Haiku 5.5 adaptive
thinking, and a Haiku 4.5 baseline for token comparison. When a new model
ships, add a check for its off form before adding it to `PROVIDER_MODELS`.
The first run (2026-10-20, Pi, anthropic SDK 1.16.0) passed all five; Haiku
5.5 cost about one seventh of Haiku 4.5 per session at similar latency,
despite about 26% more tokens for the same system prompt.

Four `ds_*` checks cover DeepSeek and need `DEEPSEEK_API_KEY`: `deepseek-flash`
thinking on, off, and an on(max)→off→on(low) toggle, plus the retired
`deepseek-v4-flash` name on the same toggle. They assert `enabled` plus a
`low|high|max` effort when on, `disabled` with no effort when off,
`reasoning_content` replay with no 400, and that the response `model` is
`deepseek-flash`. The API reports only that alias, never the snapshot behind
it, so a silent snapshot change (V4.1-Flash-1017 on 2026-10-20, per DeepSeek's
docs) shows up here only as a failed contract check. All four passed on the Pi
that day.
