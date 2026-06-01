# sys_agent

A minimal command-line agent that connects an arbitrary text prompt to a remote
LLM (OpenAI or Anthropic), gathers host facts so generated commands match the
local environment, and executes those commands only after explicit user
approval. Single Python file. No frameworks.

<!-- BEGIN EXAMPLE -->
![sys_agent example session](assets/example-session.svg)
<!-- END EXAMPLE -->

## Why

Off-the-shelf agentic CLIs (aider, claude-code, etc.) optimize for code
authoring. This one optimizes for a different loop: ad-hoc system
administration, infrastructure debugging, and "what's the right command on
*this* host?" questions where the answer depends on distro, init system,
installed package manager, and which tools happen to be on PATH.

The agent ships those host facts to the model up front so command syntax
matches reality — no more `apt` suggestions on a Mac.

## Features

- **Multi-provider**: OpenAI or Anthropic, selected at startup. Both keys
  present → interactive prompt. One key → auto-pick.
- **Host-aware prompt**: the input prompt carries the active hostname
  (`you@m3mac>`, `you@raspberrypi>`) so you always know which machine you're
  driving — no more running a Pi command on the Mac or vice versa.
- **Extended thinking** (Anthropic): optional adaptive reasoning, opt-in per
  session via `/thinking`. The scratchpad is surfaced inline; off by default
  because thinking tokens bill as output. See [Extended thinking](#extended-thinking).
- **Runtime provider/model switching**: use `/provider` and `/model` to
  inspect or change the active backend during a live REPL session.
- **Host-aware**: distro, shell, machine arch, package/init/container tool
  probes, disk context, and installed tools (`apt`/`brew`/`systemctl`/
  `docker`/etc.) are injected into the system prompt. Host facts can be
  displayed, refreshed, and expanded with `/facts`.
- **Approval-gated execution**: every proposed command is shown with its
  reason and CWD before it runs. Edit-before-run supported.
- **Local hard-deny list**: a short list of catastrophic patterns
  (`rm -rf /`, `mkfs`, fork bomb, raw `dd` to block devices) is blocked
  client-side regardless of model output or user approval.
- **Token usage display**: optional per-turn input/output token counts plus
  running session totals and percentage of context window consumed.
- **Color output**: semantic ANSI coloring with auto-detection
  (TTY/`NO_COLOR`) and runtime toggle.
- **Persistent history**: line editing and history navigation via readline
  (gnureadline on macOS). Conversational prompts persist to
  `~/.config/sys_agent/history` across sessions; meta-commands and
  short-answer prompts are excluded so Up-arrow recall stays useful.
  See [Tips & shortcuts](#tips--shortcuts) for keystrokes.
- **Audit log**: append-only JSONL record of every command the model
  proposes and its disposition (`run`/`edit`/`skip`/`deny`/`abort`) plus exit
  code, written to `~/.config/sys_agent/audit.log`. The forensic trail a 24/7
  server role needs — distinct from readline history, which stores only your
  prompts. On by default; command output bodies are excluded unless opted in.
  See [Audit log](#audit-log).
- **Zero install footprint with uv**: PEP 723 inline-script dependencies;
  `uv` handles the environment transparently.

## Requirements

- Python ≥ 3.10
- One of: `uv` (recommended) **or** pip + venv
- An OpenAI and/or Anthropic API key
- macOS/Linux for the full experience. On Windows, the stdlib lacks
  `readline` — the script still runs, but loses history persistence,
  Up/Down recall, and line-editing keystrokes.

## Install

### With uv (recommended)

```bash
git clone https://github.com/mikeoc61/sys_agent.git
cd sys_agent
chmod +x sys_agent.py
mkdir -p ~/.local/bin
ln -sf "$PWD/sys_agent.py" ~/.local/bin/sys_agent
sys_agent
```

First invocation builds a cached environment from the script's inline
dependency block; subsequent runs are instant.

> **macOS**: the inline block pulls in `gnureadline` automatically
> (`sys_platform == 'darwin'`) to replace the system libedit-backed
> readline with proper GNU readline — colored prompts render correctly
> and Up/Down history navigation redraws cleanly. Linux installs skip
> this dependency.

### With pip + venv

```bash
git clone https://github.com/mikeoc61/sys_agent.git
cd sys_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./sys_agent.py
```

`requirements.txt` carries the same `gnureadline` macOS-only marker as
the inline block.

## Configuration

API keys can come from the shell environment or from a key-value file. If
`SYS_ENV_FILE` is set, that path is used exclusively. Otherwise sys_agent
searches the following locations in priority order and uses the first that
exists:

1. `./.env` — current working directory (dotenv convention)
2. `$XDG_CONFIG_HOME/sys_agent/.env` (default `~/.config/sys_agent/.env`)
3. `~/.sys_agent.env` — home dotfile fallback

Shell-exported variables always override file values.

### Quick setup

```bash
mkdir -p ~/.config/sys_agent
cp .env.example ~/.config/sys_agent/.env
chmod 600 ~/.config/sys_agent/.env
$EDITOR ~/.config/sys_agent/.env
```

The template (`.env.example`) is committed; the real `.env` is gitignored.

### File format

Shell-style `KEY=value`, one per line. `export` prefix and `#` comments
are accepted; matching surrounding quotes are stripped.

```sh
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
# Optional overrides
SYS_PROVIDER=anthropic
SYS_OPENAI_MODEL=gpt-5.4-mini
SYS_ANTHROPIC_MODEL=claude-sonnet-4-6
```

### All environment variables

| Var | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI auth (one required) | — |
| `ANTHROPIC_API_KEY` | Anthropic auth (one required) | — |
| `SYS_PROVIDER` | Skip provider prompt: `openai` or `anthropic` | (prompt) |
| `SYS_OPENAI_MODEL` | OpenAI model string | `gpt-4o-mini` |
| `SYS_ANTHROPIC_MODEL` | Anthropic model string | `claude-haiku-4-5-20251001` |
| `SYS_THINKING` | Startup state for extended thinking: `on` / `off` (Anthropic) | `off` |
| `SYS_THINKING_EFFORT` | Adaptive thinking depth: `low`/`medium`/`high`/`xhigh`/`max` | `high` |
| `SYS_THINKING_MAX_TOKENS` | Output-token cap on thinking turns | `32000` |
| `SYS_THINKING_BUDGET` | Thinking budget for legacy models only (Haiku 4.5) | `4000` |
| `SYS_COMMAND_TIMEOUT` | Per-command wall-clock timeout, seconds | `120` |
| `SYS_AUDIT_LOG` | Audit-log path, or `off`/empty to disable | `~/.config/sys_agent/audit.log` |
| `SYS_AUDIT_BODY` | Also log command stdout/stderr in the audit log (capped) | `off` |
| `SYS_ENV_FILE` | Explicit env-file path; skips the search above | (search) |
| `SYS_COLOR` | `on` / `off` / `auto` | `auto` |
| `NO_COLOR` | If set, disables color regardless of `SYS_COLOR=auto` | — |

### Files

| Path | Purpose |
|---|---|
| `~/.config/sys_agent/.env` *(or one of the alternatives above)* | API keys and `SYS_*` overrides |
| `~/.config/sys_agent/history` | Readline history (1000-line cap, persistent across sessions) |
| `~/.config/sys_agent/audit.log` | Append-only JSONL command audit trail (path/disable via `SYS_AUDIT_LOG`) |

The history file is created on first exit. Only conversational prompts are
retained — meta-commands (`/info`, `/exit`, etc.) and short-answer prompts
(`y`/`n`, `1`/`2`) are excluded so Up-arrow recall stays useful. Clear with
`> ~/.config/sys_agent/history` if you ever want a fresh slate.

## Usage

Once running, type natural-language requests. The prompt shows the active
hostname, so it is always clear which machine will execute the command:

```
you@m3mac>       show me which services are using the most RAM
you@raspberrypi> check why bitcoind is restarting
you@raspberrypi> what's the current load average and what process is dominating?
you@m3mac>       upgrade nginx to the latest stable version
```

For mutating actions, the approval prompt is your safety net. Type `e` to edit
the command before execution.

### Tips & shortcuts

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

### Meta-commands

| Command | Effect |
|---|---|
| `/exit`, `/quit` | End session |
| `/reset` | Clear conversation history and token counters |
| `/info` | Print provider/model, session token usage, host facts |
| `/auto on\|off` | Skip approval prompt (hard-deny list still applies) |
| `/thinking on\|off` | Toggle Anthropic extended thinking (takes effect next turn) |
| `/effort [level]` | Adaptive thinking depth: `low`/`medium`/`high`/`xhigh`/`max` |
| `/tokens on\|off` | Toggle per-turn token-usage line |
| `/tokens` | Print current snapshot without changing toggle |
| `/color on\|off` | Toggle ANSI color output |
| `/audit` | Show audit-log status (enabled, path, body capture) |
| `/audit on\|off` | Toggle the command audit log at runtime |
| `/history` | Review recent command history from the audit log, paged (last 50) |
| `/history N \| all` | Show the last N entries, or the full log |
| `/provider` | Show current provider/model and available providers |
| `/provider openai\|anthropic` | Switch provider and reset conversation/token counters |
| `/model` | Show current model/context-window metadata |
| `/model MODEL_NAME` | Switch the model used by the active provider |
| `/facts` | Print current host facts |
| `/facts refresh` | Re-probe host facts and reset conversation/token counters |
| `/facts verbose on\|off` | Toggle expanded host fact collection and refresh facts |

### Runtime provider/model switching

Provider and model selection are no longer startup-only. During a session:

```text
/provider
/provider openai
/provider anthropic
/model
/model gpt-5.4-mini
/model claude-sonnet-4-6
```

Switching providers resets the active conversation and token counters because
OpenAI and Anthropic use different tool-call message formats. Host facts and
REPL toggles are preserved.

Changing the model keeps the same provider and session state. Unknown context
windows are allowed; token display falls back to absolute token counts.

### Refreshable host facts

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

### Extended thinking

Anthropic models support a reasoning pass before the model acts. It is **off by
default**: thinking tokens are billed as output (expensive on Opus), and routine
commands don't need it.

```text
/model claude-opus-4-8
/thinking on
/effort xhigh        # optional; default is high
```

Or from the environment:

```sh
SYS_THINKING=on SYS_THINKING_EFFORT=xhigh sys_agent
```

The thinking API differs by model generation, and sys_agent picks the right one
automatically:

- **Adaptive models** (Opus 4.8 / 4.7, Sonnet 4.6, Opus 4.6): the model decides
  per turn whether and how much to think. Depth is controlled by **effort**
  (`/effort`), not a token budget. Interleaved thinking is automatic. Manual
  budgets are rejected with a 400 on Opus 4.7/4.8 — sys_agent never sends them
  for these models.
- **Legacy models** (Haiku 4.5 and older): use a fixed `budget_tokens` budget
  (`SYS_THINKING_BUDGET`) plus the interleaved-thinking beta header so reasoning
  can span tool calls. Effort does not apply here.

Behavior, either way:

- Reasoning is surfaced dimmed, with a `│` margin, ahead of any proposed command
  or final answer.
- The flag is read once at the start of each turn; toggling mid-turn applies on
  the next turn (the API ignores a mid-turn toggle).
- On a thinking turn the output cap is raised to `SYS_THINKING_MAX_TOKENS`
  (default 32K) so the model has room to reason and act without truncation —
  you are only billed for tokens actually produced.
- Thinking turns are streamed internally (required by the SDK once the token
  cap is large) and buffered until complete, so a `[thinking…]` marker is shown
  while the model works; high-effort Opus turns can take a while.
- **OpenAI**: both flags are inert. Displays say so — the banner and `/help`
  show `thinking=on (inactive: openai)`, and `/info` reports `thinking_enabled`,
  `thinking_active`, and `thinking_mode`.

Reach for it on a non-obvious multi-step diagnosis (tricky `systemd`,
partitioning, networking). For everyday work, leave it off and stay on Haiku.

## Safety model

Three layers, weakest to strongest:

1. **Approval prompt** (default-on, per-command). Every `run_command` shows
   the exact shell string, the model's stated reason, and the CWD before any
   subprocess is spawned. Default answer is `y` so casual `Enter` runs it —
   read the line. Read the line.
2. **Local hard-deny list** (always-on). A short set of irrecoverable command
   patterns is blocked before the approval prompt is even shown. The model
   cannot disable this and `/auto on` cannot bypass it. Matching is
   intent-based (argv inspection through wrappers like `sudo`/`env`/`timeout`).
   See `is_denied()` and the `_DENY_*` / `_FORKBOMB_RE` tables in
   `sys_agent.py`.
3. **Command timeout** (120s wall-clock per command, configurable via
   `SYS_COMMAND_TIMEOUT`). Prevents runaway model loops from hanging the REPL
   on a single command.

The deny list is intentionally short and pattern-matched. It is **not** a
substitute for paying attention to the approval prompt. Sandbox the agent
(VM, container, `firejail`) if you want to test it on untrusted prompts.

## Audit log

A forensic record of what the agent did — **observability, not a control**. It
does not prevent anything (the approval prompt and deny list do that); it
records what was proposed and what happened, which is what you want after the
fact on a 24/7 host.

On by default. Each `run_command` the model proposes appends one JSON line to
`~/.config/sys_agent/audit.log` capturing its disposition:

| Field | When present | Meaning |
|---|---|---|
| `ts` | always | UTC timestamp, ISO-8601 (`Z`) |
| `host` / `provider` / `model` | always | active host node and backend at execution time |
| `action` | always | `run` / `edit` / `skip` / `deny` / `abort` |
| `command` | always | the command the model proposed |
| `explanation` | when given | the model's stated reason |
| `edited_command` | `action=edit` | the command as you rewrote it before running |
| `returncode` | run/edit | process exit code (`124` = timeout) |
| `truncated` | run/edit | whether output to the model was clipped at `OUTPUT_MAX_CHARS` |
| `reason` | `action=deny` | which hard-deny rule matched |
| `note` | as needed | e.g. `interrupted`, `session aborted` |
| `stdout` / `stderr` | only with `SYS_AUDIT_BODY=on` | command output, capped at `OUTPUT_MAX_CHARS` |

Output **bodies are not logged by default** — stdout/stderr can carry secrets.
Enable with `SYS_AUDIT_BODY=on` only if you accept that.

```sh
SYS_AUDIT_LOG=off sys_agent              # disable entirely
SYS_AUDIT_LOG=/var/log/sys_agent.jsonl   # custom path
SYS_AUDIT_BODY=on sys_agent              # include command output (capped)
```

At runtime: `/audit` shows status; `/audit on|off` toggles. Disposition follows
provider/model/host live, so a mid-session `/provider`, `/model`, or `/facts`
switch is reflected in subsequent records. The log is not rotated (one short
line per command); truncate with `> ~/.config/sys_agent/audit.log`.

### Reviewing history

`/history` renders the log as a numbered, human-readable list in **host-local
time** (the stored timestamps are UTC), paged through `$PAGER` (default
`less -RFX`, falling back to `more`). Day-change separators disambiguate
multi-session logs; edited commands show the form that actually ran, and
`skip`/`deny`/`abort` entries are tagged.

```text
/history          # last 50 entries (default)
/history 200      # last 200
/history all      # entire log
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

## Architecture

```
                  ┌────────────────────────────────────┐
                  │           User REPL                │
                  │  (sys_agent.py: run_repl)          │
                  └──────────────┬─────────────────────┘
                                 │
                 host_facts +    │     command output
                 conversation    │     (stdout, stderr, exit)
                                 ▼
                  ┌────────────────────────────────────┐
                  │      Provider abstraction          │
                  │  OpenAIProvider | AnthropicProvider│
                  └──────────────┬─────────────────────┘
                                 │  HTTPS + tool calling
                                 ▼
                  ┌────────────────────────────────────┐
                  │       Remote LLM API               │
                  └────────────────────────────────────┘
                                 ▲
                                 │  run_command(cmd, reason)
                                 │  ──┐
                                 │    │ approval gate
                                 │    │ deny-list check
                                 │    │ subprocess.run(...)
                                 │    │
                                 ▼    ▼
                  ┌────────────────────────────────────┐
                  │         Local host                 │
                  └────────────────────────────────────┘
```

The provider abstraction normalizes tool-call/tool-result message structure
between the two APIs (OpenAI sends one `role: tool` message per call;
Anthropic batches all `tool_result` blocks into a single user message). The
REPL is provider-agnostic.

### Runtime state

The REPL tracks active provider, active model, host facts, verbosity, token
counters, and conversation messages as mutable runtime state. `/provider`,
`/model`, and `/facts` update that state without restarting the process.

## What this is not

- Not a coding agent. Use aider, claude-code, or Cursor for that.
- Not a long-horizon autonomous agent. There is no planner, no memory beyond
  the active conversation, no parallel workers.
- Not sandboxed. Commands run as the invoking user, with that user's full
  privileges.

## License

MIT — see [LICENSE](LICENSE).
