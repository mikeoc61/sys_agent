# sys_agent

A minimal command-line agent that connects an arbitrary text prompt to a remote
LLM (OpenAI or Anthropic), gathers host facts so generated commands match the
local environment, and executes those commands only after explicit user
approval. Single Python file. No frameworks.

```
$ sys_agent
sys_agent  provider=anthropic  model=claude-haiku-4-5-20251001  host=pi5 (Linux/aarch64)
meta: /exit  /reset  /info  /auto on|off  /tokens on|off  /color on|off

you> what's eating disk on /var?

────────────────────────────────────────────────────────────────────────
COMMAND:  sudo du -sh /var/* 2>/dev/null | sort -rh | head -10
REASON:   List the largest top-level subdirectories under /var
CWD:      /home/mikeoc
────────────────────────────────────────────────────────────────────────
Run? [y]es / [n]o / [e]dit / [q]uit: y
[exit=0]
1.4G  /var/lib
820M  /var/log
...

agent> /var/lib dominates at 1.4G. Want me to break that down next?
```

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
- **Host-aware**: distro, shell, machine arch, and a probe of installed
  tools (`apt`/`brew`/`systemctl`/`docker`/etc.) are injected into the
  system prompt.
- **Approval-gated execution**: every proposed command is shown with its
  reason and CWD before it runs. Edit-before-run supported.
- **Local hard-deny list**: a short list of catastrophic patterns
  (`rm -rf /`, `mkfs`, fork bomb, raw `dd` to block devices) is blocked
  client-side regardless of model output or user approval.
- **Token usage display**: optional per-turn input/output token counts plus
  running session totals and percentage of context window consumed.
- **Color output**: semantic ANSI coloring with auto-detection
  (TTY/`NO_COLOR`) and runtime toggle.
- **Zero install footprint with uv**: PEP 723 inline-script dependencies;
  `uv` handles the environment transparently.

## Requirements

- Python ≥ 3.10
- One of: `uv` (recommended) **or** pip + venv
- An OpenAI and/or Anthropic API key

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

### With pip + venv

```bash
git clone https://github.com/mikeoc61/sys_agent.git
cd sys_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./sys_agent.py
```

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
| `SYS_ENV_FILE` | Explicit env-file path; skips the search above | (search) |
| `SYS_COLOR` | `on` / `off` / `auto` | `auto` |
| `NO_COLOR` | If set, disables color regardless of `SYS_COLOR=auto` | — |

## Usage

Once running, type natural-language requests:

```
you> show me which services are using the most RAM
you> check why bitcoind is restarting
you> what's the current load average and what process is dominating?
you> upgrade nginx to the latest stable version
```

For mutating actions, the approval prompt is your safety net. Type `e` to edit
the command before execution.

### Meta-commands

| Command | Effect |
|---|---|
| `/exit`, `/quit` | End session |
| `/reset` | Clear conversation history and token counters |
| `/info` | Print provider/model, session token usage, host facts |
| `/auto on\|off` | Skip approval prompt (hard-deny list still applies) |
| `/tokens on\|off` | Toggle per-turn token-usage line |
| `/tokens` | Print current snapshot without changing toggle |
| `/color on\|off` | Toggle ANSI color output |

## Safety model

Three layers, weakest to strongest:

1. **Approval prompt** (default-on, per-command). Every `run_command` shows
   the exact shell string, the model's stated reason, and the CWD before any
   subprocess is spawned. Default answer is `y` so casual `Enter` runs it —
   read the line. Read the line.
2. **Local hard-deny list** (always-on). A short set of irrecoverable command
   patterns is blocked before the approval prompt is even shown. The model
   cannot disable this and `/auto on` cannot bypass it. See `DENY_SUBSTRINGS`
   in `sys_agent.py`.
3. **Command timeout** (60s wall-clock per command). Prevents runaway model
   loops from hanging the REPL on a single command.

The deny list is intentionally short and pattern-matched. It is **not** a
substitute for paying attention to the approval prompt. Sandbox the agent
(VM, container, `firejail`) if you want to test it on untrusted prompts.

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

## What this is not

- Not a coding agent. Use aider, claude-code, or Cursor for that.
- Not a long-horizon autonomous agent. There is no planner, no memory beyond
  the active conversation, no parallel workers.
- Not sandboxed. Commands run as the invoking user, with that user's full
  privileges.

## License

MIT — see [LICENSE](LICENSE).
