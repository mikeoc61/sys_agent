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
