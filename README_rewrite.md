# sys_agent

A host-aware command-line AI agent for system administration, troubleshooting, and infrastructure operations.

Unlike coding-focused agents, sys_agent is optimized for understanding and operating the machine it is running on. It gathers host facts, sends those facts to a remote LLM, and executes suggested commands only after explicit user approval.

**Single Python file. No frameworks.**

![sys_agent example session](assets/example-session.svg)

---

# Quick Start

```bash
git clone https://github.com/mikeoc61/sys_agent.git
cd sys_agent
chmod +x sys_agent.py
sys_agent
```

Example:

```text
you@raspberrypi> why is bitcoind restarting?

sys_agent proposes:

systemctl status bitcoind
journalctl -u bitcoind -n 100

[y] run  [e] edit  [n] skip  [q] abort
```

Commands do not execute unless approved by the user (unless `/auto on` is enabled).

---

# Why sys_agent?

Most agentic CLIs are designed for software development. sys_agent is designed for machine operations.

| Tool | Primary Goal |
|--------|--------|
| Claude Code | Software development |
| Aider | Code editing |
| Cursor | Interactive coding |
| sys_agent | Host diagnostics and administration |

sys_agent automatically provides the model with:

- Operating system and distribution
- Shell environment
- Architecture
- Package manager
- Init system
- Available administration tools
- Running services
- Memory-heavy processes

This allows the model to generate commands appropriate for the actual host rather than guessing.

---

# Typical Workflow

```text
User Prompt
    ↓
Host Facts Injected
    ↓
LLM Analysis
    ↓
Command Proposal
    ↓
User Approval
    ↓
Command Execution
    ↓
Additional Diagnosis
```

---

# Common Use Cases

## Service Troubleshooting

```text
why is bitcoind restarting?
why did nginx fail to start?
show failed systemd services
```

## Performance Analysis

```text
what is consuming RAM?
why is load average high?
show the largest processes
```

## Storage

```text
find large log files
show disk pressure
which directories are growing?
```

## Networking

```text
what is listening on port 8333?
why can't this host reach github?
show active TCP connections
```

## Package Management

```text
upgrade nginx safely
show security updates
remove unused packages
```

---

# Core Features

## Host-Aware Prompting

Host facts are injected into the system prompt at startup.

Examples:

- Linux distribution
- macOS version
- Shell
- Architecture
- Package manager
- Container tooling
- Init system

The model begins with host-specific context rather than discovering it through trial-and-error.

## Runtime Snapshot

sys_agent captures a startup snapshot containing:

- Running services
- Top memory-consuming processes

This helps the model reference actual service and process names immediately.

Process names are transmitted without command-line arguments to reduce accidental disclosure of secrets.

## Approval-Gated Execution

Every proposed command is displayed before execution.

| Key | Action |
|------|----------|
| Enter / y | Run |
| n | Skip |
| e | Edit then run |
| q | Abort workflow |

Ctrl-C may be used to stop a workflow from anywhere within the current turn.

## Multi-Provider Support

Supported providers:

- OpenAI
- Anthropic
- DeepSeek

Provider selection can occur:

- At startup
- During an active session using `/provider`

Models may also be switched at runtime.

## Extended Thinking

Anthropic and DeepSeek support optional reasoning passes.

```text
/thinking on
/effort high
```

Useful for:

- Complex diagnostics
- Networking investigations
- Storage analysis
- Multi-step troubleshooting

Thinking is disabled by default.

## Audit Logging

Every proposed command may be recorded in an append-only JSONL audit log.

Actions tracked include:

- run
- edit
- skip
- deny
- abort

Command output logging remains optional.

## Token Visibility

Optional per-turn display:

- Input tokens
- Output tokens
- Session totals
- Context-window consumption

## Command History

Persistent readline history:

```text
↑ / ↓
Ctrl-R
Ctrl-A
Ctrl-E
```

Conversational prompts persist across sessions.

## UX Features

- ANSI color support
- Activity spinner
- Runtime configuration toggles
- Persistent history
- Provider switching

---

# Requirements

- Python 3.10+
- uv (recommended) or pip+venv
- At least one provider API key
- macOS or Linux recommended

Windows works but lacks full readline functionality.

---

# Installation

## Using uv (Recommended)

```bash
git clone https://github.com/mikeoc61/sys_agent.git
cd sys_agent

chmod +x sys_agent.py

mkdir -p ~/.local/bin
ln -sf "$PWD/sys_agent.py" ~/.local/bin/sys_agent

sys_agent
```

The first launch builds a cached environment from the script's inline dependency block.

## Using pip + venv

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

./sys_agent.py
```

---

# Configuration

## API Keys

Provide one or more:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
```

## Environment File Search Order

1. `./.env`
2. `$XDG_CONFIG_HOME/sys_agent/.env`
3. `~/.sys_agent.env`

Shell-exported variables override file values.

## Quick Setup

```bash
mkdir -p ~/.config/sys_agent
cp .env.example ~/.config/sys_agent/.env
chmod 600 ~/.config/sys_agent/.env
$EDITOR ~/.config/sys_agent/.env
```

## Frequently Used Environment Variables

| Variable | Purpose |
|-----------|----------|
| SYS_PROVIDER | Default provider |
| SYS_OPENAI_MODEL | OpenAI model |
| SYS_ANTHROPIC_MODEL | Anthropic model |
| SYS_DEEPSEEK_MODEL | DeepSeek model |
| SYS_THINKING | Enable startup thinking |
| SYS_COMMAND_TIMEOUT | Command timeout |
| SYS_AUDIT_LOG | Audit log path |
| SYS_COLOR | Color mode |
| SYS_PROGRESS | Activity indicator |

---

# Command Cheat Sheet

```text
/facts
/facts refresh
/provider
/model
/history
/thinking on
/tokens
/reset
/exit
```

---

# Command Reference

| Command | Description |
|----------|-------------|
| /facts | Show host facts |
| /facts refresh | Re-probe host facts |
| /facts verbose on/off | Toggle expanded host facts |
| /provider | Show/change provider |
| /model | Show/change model |
| /thinking on/off | Toggle reasoning |
| /effort | Configure reasoning depth |
| /tokens | Show token usage |
| /audit | Show audit status |
| /history | Review audit history |
| /reset | Reset conversation |
| /exit | Exit session |

---

# Safety Model

Three layers of protection:

## 1. Approval Prompt

Every command is shown before execution.

## 2. Hard Deny List

Blocks catastrophic patterns such as:

- rm -rf /
- mkfs
- fork bombs
- raw block-device writes

The deny list cannot be bypassed by model output.

## 3. Command Timeout

Commands are terminated after a configurable wall-clock timeout.

The approval prompt remains the primary safety mechanism.

---

# Runtime Snapshots

The runtime snapshot contains:

## running_services

Active services discovered from:

- systemd (Linux)
- launchctl (macOS)

## top_processes

Top memory-consuming processes aggregated by name.

Example:

```text
bitcoind
gunicorn ×8
postgres
```

Snapshots are collected at startup and refreshed using:

```text
/facts refresh
```

---

# Audit Log

Default location:

```text
~/.config/sys_agent/audit.log
```

Records include:

- timestamp
- provider
- model
- action
- command
- return code

Useful for:

- Forensics
- Change tracking
- Operational review

---

# Architecture

```text
                User
                 │
                 ▼
         sys_agent REPL
                 │
                 ▼
        Provider Abstraction
                 │
                 ▼
            Remote LLM
                 │
                 ▼
            Local Host
```

Providers normalize differences between:

- OpenAI
- Anthropic
- DeepSeek

The REPL remains provider-agnostic.

---

# What This Is Not

- Not a coding agent
- Not a planner
- Not a long-horizon autonomous system
- Not sandboxed

Commands run with the permissions of the invoking user.

---

# License

MIT
