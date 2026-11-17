# Model notes

[Getting started](../README.md) · [Configuration](configuration.md) ·
[Advanced usage](advanced-usage.md) · [Technical reference](technical-reference.md)

This page records why sys_agent's model lists are what they are: each model's
price, the probe evidence behind each listed model, and why other models are
unlisted or removed. The lists themselves live in the PROVIDER & MODEL CONFIG
section of `sys_agent.py` (`DEFAULT_*_MODEL`, `PROVIDER_MODELS`,
`CONTEXT_WINDOWS`, `_OPENAI_REASONING_EFFORTS`), with the per-model thinking
tables just below it. For advice on choosing a model, see
[OpenAI models](advanced-usage.md#openai-models) and
[extended thinking](advanced-usage.md#extended-thinking).

Prices are per 1M tokens, input/output. A date records when a fact was checked
or a probe was run.

Each model has one of three statuses:

- **Listed:** in `PROVIDER_MODELS`, so `/model` suggests it and switches to it
  silently.
- **Unlisted:** not in `PROVIDER_MODELS`, but its `CONTEXT_WINDOWS` entry and
  any thinking or effort table rows are kept. A `SYS_*_MODEL` setting or a
  `/model` switch still selects it, with an unlisted-model warning, and its
  context-% and thinking state display correctly.
- **Removed:** no entries at all, because the model no longer works.

## OpenAI

Checked November 2026. Every GPT-6 and GPT-6.1 model has a 1.05M-token context
and 128K max output; above 272K input tokens, the input rate doubles and the
output rate is 1.5x.

### Listed, in picker order

- **`gpt-6-luna`**: $0.10/$0.50, the GPT-6 low tier. Default since 2026-11-16.
  - 2026-09-23 probe (Chat Completions): a tool call, its replay and the final
    answer round-trip at effort `none`.
  - 2026-11-16 probe (Pi, real system prompt, a three-prompt canned session run
    twice, Chat Completions at effort `none`): correct answers on both runs at
    about a seventh of `gpt-5.4-mini`'s cost. One run chained commands with
    `&&` against the one-step-per-call rule, so its discipline is weaker than
    mini's. The evidence is three easy read-only prompts, nothing multi-step.
  - On the Responses API it reasons while calling tools (79 reasoning tokens at
    `high`, 2026-11-16).
- **`gpt-6.1-terra`**: $1/$5, cached input $0.05 (docs, 2026-11-16; knowledge
  cutoff 2026-06-01). The GPT-6.1 mid tier and the middle pick.
- **`gpt-6-sol`**: $2/$10. The GPT-6 tier above Luna (GPT-6 had no Terra; in
  GPT-5.6, Sol was the top tier). Not deprecated (deprecations page,
  2026-11-16).
- **`gpt-6.1-sol`**: $2/$10, cached input $0.10. The same price as `gpt-6-sol`,
  one generation newer. Always reasons: it rejects effort `none` and `minimal`.
  Its tools work only on the Responses API, so it could not be used here until
  v1.31.0 moved OpenAI to that API.
- **`gpt-5.4-mini`**: $0.75/$4.50, 400K context. Default until 2026-11-16, kept
  listed as the fallback if a newer model falls short. No deprecation entry
  (2026-11-16).

### Why the Responses API

On Chat Completions, GPT-5.6 and every GPT-6 model except `gpt-6-astra` and
`gpt-6.1-sol` accept function tools only at reasoning effort `none`, so they ran
here as non-reasoning models. `gpt-6-astra` and `gpt-6.1-sol` reject tools at
every effort, `none` included (probes 2026-09-23 and 2026-11-16). OpenAI's
reasoning guide states the second limit too. On the Responses API every listed
model calls tools while reasoning. The effort grades each model accepts there
(`_OPENAI_REASONING_EFFORTS`) come from each model's error for an unsupported
value (2026-11-16); no model accepts `minimal`.

### Unlisted

- **`gpt-6-astra`**: the flagship, $10/$50. Its price alone puts it in the
  `claude-fable-5` tier, the wrong tier for routine host work. It works on the
  Responses path and always reasons.
- **`gpt-4o-mini`**: $0.15/$0.60. It costs more than `gpt-6-luna` on both input
  and output, and it is the chattiest tier: with `tool_choice=auto` it tends to
  ask ("Would you like me to check…?") instead of calling `run_command`, which
  stalls multi-step investigations. It is not a reasoning model: the Responses
  API rejects `reasoning.effort` for it, so it gets no reasoning field.
- **`gpt-5.6-luna`**: $0.20/$1.20. `gpt-6-luna` is cheaper, with the same
  constraints.
- **`gpt-5.6-terra`**: $2/$12. `gpt-6-sol` matches its input price with
  cheaper output.
- **`gpt-5.6-sol`**: $5/$30. Never listed (the wrong tier for this tool).

### Removed

- **`gpt-4.1-nano`**: removed 2026-11-11, list entry, window and all. OpenAI
  shut it down on 2026-10-23 (deprecations page), so no setting can make it
  work, and a window entry would only make a dead model look supported.
  `gpt-6-luna` is the low-cost pick ($0.10/$0.50 against nano's $0.10/$0.40).

## Anthropic

Checked October 2026; Fable 5.2 in November 2026. The per-model thinking-off
forms are in `_THINKING_OFF_PAYLOAD` and the models that cannot turn thinking
off are in `_THINKING_ALWAYS_ON_MODELS`; the comments there explain how
sys_agent handles each.

### Listed

- **`claude-haiku-5-5`**: $0.10/$0.50 with cache reads at $0.01, for prompts up
  to 100K tokens; $0.50/$2.50 above that (the whole request, cache tokens
  counted). In the API since 2026-10-07. Default since 2026-11-11: the
  `haiku55_off` and `haiku55_on` live probes passed on the Pi on 2026-10-20 and
  again on 2026-11-11, at about one seventh of Haiku 4.5's cost per session.
  - Adaptive thinking runs by default when no thinking field is sent.
    `{"type":"disabled"}` is accepted at effort high or below (thinking docs,
    October 2026). The server default effort is medium.
  - Accepts a forced `tool_choice`; rejects prefill and sampling parameters
    (sys_agent uses neither).
- **`claude-sonnet-5-5`**: $2/$10, cache reads $0.10 (0.05x; Sonnet 5 charged
  $0.20). In the API since 2026-09-28.
  - Adaptive thinking runs by default, and `{"type":"disabled"}` is rejected.
    Its lowest setting is `{"type":"between_tools"}`: no up-front thinking, but
    notes written between tool calls still come back as thinking blocks with
    summary text, shown dimmed like any thinking. That setting is accepted at
    effort high or below and takes no other field; sys_agent's off branch sends
    no effort, so the server default (high) applies. In the Models API
    (2026-10-15), `capabilities.thinking.types.between_tools` is true only for
    this model.
  - Rejects a forced `tool_choice` (sys_agent sends `auto`).
- **`claude-opus-5-5`**: $4/$20, cache reads $0.20 (0.05x). The current Opus, in
  the API since 2026-09-21.
  - Thinking is always on: `{"type":"disabled"}` is rejected at every effort
    (probe 2026-09-24). The server default effort is medium (Opus 5: high).
  - A probe with sys_agent's thinking-off request shape (no thinking field)
    gave clean `run_command` calls and 169–490 output tokens per turn over a
    four-turn session. Empty thinking blocks (display "omitted") replay
    without error.
  - Rejects a forced `tool_choice`.

### Unlisted

- **`claude-haiku-4-5-20251001`**: $1/$5. The default until 2026-11-11, when
  Haiku 5.5 replaced it at a tenth of the price. It uses the legacy
  enabled-plus-`budget_tokens` thinking form and has a 200K window.
- **`claude-sonnet-5`**: $2/$10. Sonnet 5.5 has the same price and half the
  cache-read rate (unlisted 2026-10-20). Thinking is on by default; the off
  form is `{"type":"disabled"}`.
- **`claude-opus-5`**: $5/$25. Opus 5.5 is cheaper and newer. Thinking is on by
  default but can be disabled, so its off form is kept for anyone who sets it.
  Opus 5 rejects `disabled` at effort xhigh or max; sys_agent never hits that,
  because its off branch sends no effort and the server default (high)
  applies. With thinking off it can write a proposed command as text instead
  of a tool call.
- **`claude-opus-4-8`**: $5/$25, thinking off by default. Opus 5.5 beats it on
  price.
- **`claude-opus-4-7`**: $5/$25, two generations old.
- **`claude-sonnet-4-6`**: $3/$15. Sonnet 5.5 is cheaper. Still active
  upstream, with retirement not before 2027-02-17.

Never listed, because they are the wrong tier for this tool (like
`gpt-6-astra`):

- **`claude-fable-5`, `claude-fable-5-1`**: $10/$50, thinking always on. Fable
  5.1 (in the API since 2026-08-28) also rejects a forced `tool_choice`.
- **`claude-fable-5-2`**: $8/$40, cache reads $0.20 (in the API since
  2026-11-09). The same surface as Fable 5.1: thinking always on, server
  default effort high, forced `tool_choice` rejected.
  - It checks replayed thinking blocks for edited history on every account
    (Fable 5.1 only for new accounts). sys_agent's REPL passes that check: its
    only edits remove whole turns from the end, and a system prompt change
    resets the conversation.
  - 2026-11-11 probe (Pi, the `fable52_*` live probes): with no thinking field,
    tool turns carry thinking blocks, and they replay without error across an
    on/off toggle and after a turn rollback.

## DeepSeek

Checked October 2026. DeepSeek prices are off-peak/peak; peak is 01:00–04:00
and 06:00–10:00 UTC, Monday to Friday. Thinking is on by default server-side
(probe 2026-09-10: with no thinking field, `reasoning_content` and
`reasoning_tokens` come back), so sys_agent sends an explicit `disabled` when
`/thinking` is off. Effort grades are low, high and max.

### Listed

- **`deepseek-flash`**: the default. DeepSeek-V4.1-Flash (released
  2026-09-10), a new causal encoder-decoder architecture with 8B/16B active
  parameters (input/output). 1M context, 384K max output, thinking and
  non-thinking modes, tool calling, and native vision (unused here). Input
  $0.15/$0.30 (cache hit $0.003/$0.006), output $0.60/$1.20.
  - An unversioned alias that tracks DeepSeek's current Flash. Since
    2026-10-20 it is served by DeepSeek-V4.1-Flash-1017, an open-weights minor
    update with an unchanged calling method, per DeepSeek's docs. The API
    reports only the alias, so this comes from the docs and no probe can
    confirm it.
  - Replay and thinking on, off and toggled were verified on the Pi with the
    `ds_*` live probes.
- **`deepseek-v4-pro`**: DeepSeek-V4-Pro-0813 (GA 2026-08-13). Input
  $0.66/$1.32 (cache hit $0.022/$0.044), output $1.98/$3.96. 1M context, 384K
  max output, thinking on by default, effort low, high or max (default high).
  The stronger DeepSeek option, though older than Flash.
  - Not retired. The 2026-09-10 release notice announced routing it to
    V4.1-Flash from 2026-09-14, but the changelog reversed that the same day,
    citing user demand: service continues and billing is unchanged. The news
    page still shows the routing notice, so the API is the authority: on
    2026-09-24 the response `model` was `deepseek-v4-pro`, and a four-turn
    tool session replayed `reasoning_content` cleanly with thinking on and off.
  - Expect this name to move again when V4.1-Pro ships.

### Unlisted

- **`deepseek-v4-flash`**: retired. The server routes it to V4.1-Flash-1017 and
  bills the Flash price (probe 2026-10-20: the response `model` reports
  `deepseek-flash`). Its window entry is kept for existing
  `SYS_DEEPSEEK_MODEL` settings.

## Updating this page

When a model is added, unlisted, removed or made the default, update its entry
here in the same commit as the change to `sys_agent.py`, and update the
user-facing [OpenAI models](advanced-usage.md#openai-models) table when the
OpenAI list changes. Probe a model before listing it: see
[live contract probe](model-monitoring.md#live-contract-probe).
