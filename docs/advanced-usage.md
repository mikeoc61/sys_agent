# Advanced usage

[Getting started](../README.md) · [Configuration](configuration.md) ·
[Advanced usage](advanced-usage.md) ·
[Technical reference](technical-reference.md)

Use these options after your first session. Commands starting with `/` are
entered inside sys_agent. Shell examples using `sys_agent` assume you installed
the [terminal shortcut](configuration.md#install-a-terminal-shortcut).

## Meta-commands

| Command | Effect |
|---|---|
| `/exit`, `/quit` | End the session (these and Ctrl-D are the only ways to quit) |
| `/reset` | Clear conversation history and token counters |
| `/info` | Print provider/model, session token usage, host facts |
| `/auto on\|off` | Skip approval prompt (hard-deny list still applies) |
| `/thinking on\|off` | Toggle extended thinking — Anthropic / DeepSeek (takes effect next turn) |
| `/effort [level]` | Thinking depth — Anthropic adaptive (all 5), DeepSeek (`low`/`high`/`max`); shows the effective grade; needs `/thinking on` |
| `/tokens on\|off` | Toggle per-turn token-usage line |
| `/tokens` | Print current snapshot without changing toggle |
| `/color on\|off` | Toggle ANSI color output |
| `/audit` | Show audit-log status (enabled, path, body capture) |
| `/audit on\|off` | Toggle the command audit log at runtime |
| `/history` | Review recent command history from the audit log, paged (last 50) |
| `/history N \| all` | Show the last N entries, or the full log |
| `/consult` | Second opinion: ask the other configured providers how they'd open the current question (executes nothing) |
| `/consult --fresh` | Same, but consult the question alone — no prior conversation history |
| `/provider` | Show current provider/model and available providers |
| `/provider openai\|anthropic\|deepseek` | Switch provider and reset conversation/token counters |
| `/model` | Show current model/context-window metadata |
| `/model MODEL_NAME` | Switch the model used by the active provider |
| `/facts` | Print current host facts |
| `/facts refresh` | Re-probe host facts and reset conversation/token counters |
| `/facts verbose on\|off` | Toggle expanded host fact collection and refresh facts |

## Runtime provider/model switching

Provider and model selection are no longer startup-only. During a session:

```text
/provider
/provider openai
/provider anthropic
/provider deepseek
/model
/model gpt-5.4-mini
/model claude-sonnet-5
/model deepseek-flash
```

Switching providers resets the active conversation and token counters because
the providers use different tool-call message formats (Anthropic batches
`tool_result` blocks; OpenAI and DeepSeek send one `role: tool` message per
call). Host facts and REPL toggles are preserved.

Changing the model keeps the same provider and session state. Unknown context
windows are allowed; token display falls back to absolute token counts.

### Unsupported: OpenAI GPT-6 Astra

`gpt-6-astra` cannot be used with sys_agent (checked 2026-09-23). sys_agent
talks to OpenAI through the Chat Completions endpoint, and there Astra rejects
function tools, which is how every command is proposed. The error message tells
you to set `reasoning_effort` to `none`, but Astra rejects `none` too. Only
`low`, `medium`, `high`, and `xhigh` are accepted, and all of them hit the same
tools error. Leaving the field out hits it as well. Astra with tools works only
on OpenAI's Responses API, which sys_agent does not use.

`/model gpt-6-astra` is still accepted (it matches the `gpt-` prefix), but the
first question fails with a 400 error and no command is run. sys_agent replaces
OpenAI's misleading "set reasoning_effort to 'none'" advice with an explanation
that links here. Switch back with `/model gpt-5.4-mini`, or pick a listed
model with `/models`. Astra is also priced at $10/$50 per million input/output
tokens, the same tier as `claude-fable-5`, which sys_agent leaves out of its
model list as too expensive for this workload.

The rest of the GPT-6 family *does* work and is in the model list:
`gpt-6-luna` ($0.10/$0.50) is the low-cost option, and `gpt-6-sol` ($2/$10) is
the mid tier. GPT-6 renamed its tiers: Luna, then Sol, then Astra at the top,
with no Terra. Both accept `reasoning_effort` `none`, and sys_agent sends it
automatically, so both run without reasoning here. The default stays
`gpt-5.4-mini` until the GPT-6 models' tool use holds up in real sessions.

`gpt-4o-mini`, `gpt-5.6-luna`, and `gpt-5.6-terra` were removed from the model
list on 2026-09-23. Each costs the same as or more than a GPT-6 replacement.
A `SYS_OPENAI_MODEL` setting or `/model` switch that names one of them still
works, with an unlisted-model warning.

On the Anthropic side, `claude-opus-5-5` ($4/$20) replaced `claude-opus-5` and
`claude-opus-4-8` (both $5/$25) in the model list on 2026-09-24. Opus 5.5 can't
turn thinking off (see [extended thinking](#extended-thinking)). Existing
`SYS_ANTHROPIC_MODEL` settings that name the older models keep working with the
same warning. `claude-fable-5-1` is recorded but not listed: at $10/$50 it is
priced for harder work than this tool needs.

## Multi-provider consult

`/consult` asks the *other* configured providers how they would approach the
current question and shows their opening move beside the active provider's — a
second opinion at a decision point. It is **read-only: nothing it returns is
executed**, and it never touches the approval gate. The point is not to vote or
to auto-pick a winner; it is to surface agreement or disagreement so you decide.

```text
/consult
/consult --fresh
```

Requires at least one *other* provider key configured — with a single key there
is nothing to consult.

**What you see.** Each consulted provider works the question independently from
the same context the active provider had, and — because consult executes
nothing — you see only its *opening turn*: the first command (or several) it
would propose, or a direct answer. That opening is compared against the active
provider's own first move, shown as the `reference`. An opening turn may bundle
several commands; in the REPL those are still approved and run one at a time, so
the count reflects how many approvals that opening would cost you, not
concurrency.

**Convergence / divergence.** When the consulted providers propose the same
command(s) it is flagged as convergence; when they don't, as divergence. The
comparison is deliberately *literal* — it never claims two differently-worded
commands are equivalent, since overstating sameness is the failure mode to
avoid. Read divergence as a prompt to compare approaches, weighted by stakes:
on an interchangeable read-only probe it is low-signal; on a non-obvious
diagnosis — where one provider refreshes state before reading and another does
not, say — it is exactly the second opinion worth having before you commit.

**Second opinion before running anything.** Abort a turn with `q` (or Ctrl-C)
*before* approving its first command, then `/consult`: the aborted question —
not the previous turn — is what gets consulted. This is the cleanest use of the
feature: see how every provider would open, with zero side effects, then decide
what to actually run.

**`--fresh`.** By default consult carries the prior conversation as context
(apples-to-apples with what the active provider saw). On a context switch — a
new, self-contained question where the accumulated session is just noise —
`/consult --fresh` consults the question alone (question + host facts, no prior
turns), which is also cheaper. It applies on every path: aborting only drops the
one aborted turn, not the earlier session, so the abort and skip paths carry the
same history without it.

Consult queries the providers concurrently and reports its token cost
*separately* from the session counters, which stay tied to the active
conversation's context-window accounting. The session's thinking and effort
settings carry into the consult call, but the *model* does not: each consulted
provider is built fresh and runs its own configured default model, since a
model name is provider-specific. The active provider's first move is shown as
the reference.

## Refreshable host facts

```text
/facts
/facts refresh
/facts verbose on
/facts verbose off
```

`/facts` prints the currently injected host metadata. `/facts refresh`
re-probes the machine and rebuilds the system prompt. Verbose mode adds
additional environment details such as CPU count, PATH, basic container
detection, and network-tool availability. Refreshing host facts resets the
active conversation and token counters so the model receives a clean, current
system prompt.

## Extended thinking

Anthropic and DeepSeek models support a reasoning pass before the model acts.
It is **off by default**: thinking tokens are billed as output (expensive on
Opus), and routine commands don't need it.

```text
/model claude-opus-5-5
/thinking on
/effort xhigh
```

DeepSeek works the same way — `/thinking on` with `deepseek-flash`.

Or from the environment:

```sh
SYS_THINKING=on SYS_THINKING_EFFORT=xhigh sys_agent
```

The thinking API differs by model/provider, and sys_agent picks the right one
automatically:

- **Anthropic adaptive** (Opus 5 / 4.8 / 4.7, Sonnet 5 / 4.6, Opus 4.6): the
  model decides per turn whether and how much to think. Depth is controlled by
  **effort** (`/effort`), not a token budget. Interleaved thinking is automatic.
  Manual budgets are rejected with a 400 on Opus 4.7/4.8 — sys_agent never sends
  them for these models. **Sonnet 5 and Opus 5 think by default** even when no
  thinking field is sent, so with `/thinking off` sys_agent sends an explicit
  `disabled` to keep the off-by-default cost rationale intact. On Opus 5 that
  `disabled` is sent without an effort field (the API rejects `disabled` at
  effort `xhigh`/`max`); with thinking disabled Opus 5 may occasionally phrase a
  proposed command as text rather than a tool call, so use `/thinking on` if you
  hit that.
- **Anthropic always-on** (Opus 5.5, Fable 5 / 5.1): these models cannot turn
  thinking off; the API rejects `disabled` with a 400. `/thinking off` therefore
  can't be honored. It only stops sys_agent from sending an effort level, so
  the server default applies (`medium` on Opus 5.5). Displays say so: the state
  reads `on (always on for claude-opus-5-5)` instead of `off`. Because the model
  may still think, sys_agent uses the thinking-turn output cap and streaming for
  these models even when thinking is off. `/thinking on` sends your `/effort`
  level as usual, so on Opus 5.5 `/thinking on` with `/effort low` can cost
  *less* than "off", which runs at `medium`.
- **Anthropic legacy** (Haiku 4.5 and older): use a fixed `budget_tokens` budget
  (`SYS_THINKING_BUDGET`) plus the interleaved-thinking beta header so reasoning
  can span tool calls. Effort does not apply here.
- **DeepSeek** (`deepseek-flash` = V4.1 Flash): thinking is set per turn via the
  request body. **DeepSeek thinks by default** when no thinking field is sent,
  so with `/thinking off` sys_agent sends an explicit `disabled` (same rationale
  as Sonnet 5 / Opus 5). Depth maps from **effort** to DeepSeek's three grades:
  `low` → `low`, `medium`/`high` → `high`, `xhigh`/`max` → `max` (levels without
  a native grade round up; DeepSeek's own server table maps `xhigh` → `high`).
  No token budget or max-tokens raise applies. The legacy `deepseek-v4-flash`
  and `deepseek-v4-pro` names are accepted but server-routed to V4.1 Flash
  (`v4-pro` from 2026-09-14, until V4.1 Pro ships).
  DeepSeek imposes a strict replay contract: once a thinking turn makes a tool
  call, the reasoning scratchpad must be echoed back on every subsequent
  assistant message or the next request 400s. sys_agent handles this internally
  by preserving `reasoning_content` on each assistant turn, so multi-turn tool
  use just works.

Behavior, all paths:

- Reasoning is surfaced dimmed, with a `│` margin, ahead of any proposed command
  or final answer.
- The flag is read once at the start of each turn; toggling mid-turn applies on
  the next turn (the API ignores a mid-turn toggle).
- A `[thinking…]` marker is shown while the model works; high-effort Opus turns
  can take a while.
- `/effort` echoes the grade actually used, not just what you typed: on DeepSeek
  it shows the mapped grade with a `(rounded up to DeepSeek grade)` note when
  the level has no native grade (`medium` → `high`, `xhigh` → `max`); on
  providers/models that ignore effort it says `(no effect on this
  provider/model)`; and it appends `needs /thinking on` when effort is set but
  thinking is off. The stored level is your request, so switching to a
  finer-grained provider still honors it.

Anthropic-specific:

- On a thinking turn, and on every turn for the always-on models, the output
  cap is raised to `SYS_THINKING_MAX_TOKENS` (default 32K) so the model has room to reason and act without truncation —
  you are only billed for tokens actually produced. (DeepSeek needs no such
  raise.)
- Thinking turns are streamed internally (required by the SDK once the token
  cap is large) and buffered until complete. DeepSeek uses a plain
  non-streaming call.

**OpenAI**: both flags are inert. Displays say so — the banner and `/help`
show `thinking=on (inactive: openai)`, and `/info` reports `thinking_enabled`,
`thinking_active`, and `thinking_mode`.

Reach for it on a non-obvious multi-step diagnosis (tricky `systemd`,
partitioning, networking). For everyday work, leave it off and stay on a fast
tier (Haiku, Flash).

## Line editing and history

Line editing is provided by readline (or gnureadline on macOS, installed
automatically).

| Key | Action |
|---|---|
| Up / Down | Cycle through prior conversational prompts |
| Ctrl-R | Reverse-incremental search through history |
| Ctrl-A / Ctrl-E | Jump to start / end of line |
| Ctrl-W | Delete previous word |
| Ctrl-U / Ctrl-K | Delete to start / end of line |
| Tab | (No completion — sys_agent doesn't bind any) |

History persists across sessions; recall surfaces only real prompts, so
Up-arrow won't waste your time on `y`/`n` answers or meta-commands. To
start with a clean slate: `> ~/.config/sys_agent/history`.

## Multi-line input

Each Enter submits a message, so a pasted multi-line block would otherwise
become one conversational turn per line. Wrap it in `"""` delimiters to send it
as a single message:

| Input | Effect |
|---|---|
| A line starting with `"""` | Opens a block; any text after the `"""` is its first line |
| A line ending with `"""` | Closes the block and sends it; text before the `"""` is kept |
| `"""text"""` on one line | Sends `text` |
| Ctrl-D inside a block | Closes the block and sends it |
| Ctrl-C inside a block | Discards the block and returns to the prompt |

Continuation lines are prompted with `...` and keep their indentation. A
block is always a message to the AI: text starting with `/` inside it is not
run as a meta-command. It is saved to history as a single line, so Up-arrow
recalls it as one line that you can edit and send again.

This is plain line reading, not terminal bracketed paste, so it behaves the
same on GNU readline and libedit. Limitation: a pasted line that itself ends
in `"""`, such as a Python docstring, closes the block early.

## Review command history

`/history` renders the log as a numbered, human-readable list in **host-local
time** (the stored timestamps are UTC), paged through `$PAGER` (default
`less -RFX`, falling back to `more`). Day-change separators disambiguate
multi-session logs; edited commands show the form that actually ran, and
`skip`/`deny`/`abort` entries are tagged.

```text
/history
/history 200
/history all
```

```
# audit history — last 50 of 312 records  (host-local time)
── 2026-05-31 ──
 1.  17:11:36  systemctl status bitcoind   — Checked bitcoind service status
 2.  17:11:43  tail -100 debug.log | head -50 [edited]   — Reviewed log entries
 3.  17:11:49  bitcoin-cli getblockchaininfo (exit 1)   — Got blockchain status
 4.  17:16:29  rm -rf / [denied: recursive delete targeting the filesystem root]
```

For ad-hoc queries on the raw JSONL, `jq` is still the sharper tool:

```sh
jq -r 'select(.action=="run") | "\(.ts) [\(.returncode)] \(.command)"' \
  ~/.config/sys_agent/audit.log
jq 'select(.action=="deny")' ~/.config/sys_agent/audit.log
```
