# Configuration reference

[Getting started](../README.md) · [Configuration](configuration.md) ·
[Advanced usage](advanced-usage.md) ·
[Technical reference](technical-reference.md)

Settings are optional once you have added an API key. This page covers the full
configuration and alternative installation methods.

## Configuration

API keys can come from the shell environment or from a key-value file. If
`SYS_ENV_FILE` is set, that path is used exclusively. Otherwise sys_agent
searches the following locations in priority order and uses the first that
exists:

1. `./.env` — current working directory (dotenv convention)
2. `$XDG_CONFIG_HOME/sys_agent/.env` (default `~/.config/sys_agent/.env`)
3. `~/.sys_agent.env` — home dotfile fallback

Shell-exported variables always override file values. Every `SYS_*` setting
is resolved *after* the file is loaded, so values set there take effect
identically to shell-exported ones. A malformed value (a non-numeric timeout,
an unknown effort grade) is reported as a `[config]` warning at startup and
the current value (normally the built-in default) is kept, rather than aborting
startup. That warn-and-continue handling covers the numeric settings
(`SYS_COMMAND_TIMEOUT`, `SYS_THINKING_BUDGET`, `SYS_THINKING_MAX_TOKENS`,
`SYS_TOP_PROCESSES`), the on/off switches (`SYS_THINKING`, `SYS_DISCLAIMER`,
`SYS_PROGRESS`, `SYS_UPDATE_CHECK`), and `SYS_THINKING_EFFORT`.

Two settings behave differently. `SYS_PROVIDER` is validated separately and
**does** exit with an error message: an unrecognized provider name, or a valid
one whose API key is not set, stops startup rather than falling back. The
`SYS_*_MODEL` strings are not validated here at all — pinning an unlisted model
is a supported escape hatch, so a typo in one surfaces later as a warning from
`/model` or as an API error, not as a `[config]` line.

### Quick setup

```bash
mkdir -p ~/.config/sys_agent
cp .env.example ~/.config/sys_agent/.env
chmod 600 ~/.config/sys_agent/.env
nano ~/.config/sys_agent/.env
```

Run the copy command from the project folder. This writes the same file the
[getting-started steps](../README.md#3-add-your-api-key) create by hand, so use
one path or the other, not both. Keep only the API-key lines you use and replace
their placeholders; delete unused key lines. In nano, save with Ctrl-O, Enter,
then exit with Ctrl-X. The template is tracked in Git; the real `.env` is
ignored.

### File format

Shell-style `KEY=value`, one per line. `export` prefix and `#` comments
are accepted; matching surrounding quotes are stripped.

```sh
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
# Optional overrides
SYS_PROVIDER=anthropic
SYS_OPENAI_MODEL=gpt-6-luna
SYS_ANTHROPIC_MODEL=claude-sonnet-5
SYS_DEEPSEEK_MODEL=deepseek-flash
```

### All environment variables

| Var | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI auth (one of the three required) | — |
| `ANTHROPIC_API_KEY` | Anthropic auth (one of the three required) | — |
| `DEEPSEEK_API_KEY` | DeepSeek auth (one of the three required) | — |
| `SYS_PROVIDER` | Skip provider prompt: `openai`, `anthropic`, or `deepseek` | (prompt) |
| `SYS_OPENAI_MODEL` | OpenAI model string (default favors reliable tool use; set `gpt-6-luna` for lower cost) | `gpt-5.4-mini` |
| `SYS_ANTHROPIC_MODEL` | Anthropic model string | `claude-haiku-4-5-20251001` |
| `SYS_DEEPSEEK_MODEL` | DeepSeek model string | `deepseek-flash` |
| `SYS_THINKING` | Startup state for extended thinking: `on` / `off` (Anthropic / DeepSeek) | `off` |
| `SYS_THINKING_EFFORT` | Thinking depth: `low`/`medium`/`high`/`xhigh`/`max` (DeepSeek rounds up to `low`/`high`/`max`) | `high` |
| `SYS_THINKING_MAX_TOKENS` | Output-token cap on thinking turns (Anthropic only) | `32000` |
| `SYS_THINKING_BUDGET` | Thinking budget for legacy Anthropic models only (Haiku 4.5) | `4000` |
| `SYS_COMMAND_TIMEOUT` | Per-command wall-clock timeout, seconds | `120` |
| `SYS_TOP_PROCESSES` | Top-N processes (by RSS) in the runtime snapshot | `10` |
| `SYS_AUDIT_LOG` | Audit-log path, or `off`/empty to disable | `~/.config/sys_agent/audit.log` |
| `SYS_AUDIT_BODY` | Also log command stdout/stderr in the audit log (capped) | `off` |
| `SYS_ENV_FILE` | Explicit env-file path; skips the search above | (search) |
| `SYS_COLOR` | `on` / `off` / `auto` | `auto` |
| `NO_COLOR` | If set, disables color regardless of `SYS_COLOR=auto` | — |
| `SYS_PROGRESS` | Activity spinner during model calls / command execution: `on`/`off` | `on` |
| `SYS_DISCLAIMER` | Startup notice that model-proposed commands can be wrong: `on`/`off` | `on` |
| `SYS_UPDATE_CHECK` | Check at startup whether `sys_agent.py` matches GitHub `main` ([details](technical-reference.md#update-check)): `on`/`off` | `on` |

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

## Alternative installation: pip and venv

```bash
git clone https://github.com/mikeoc61/sys_agent.git
cd sys_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python3 sys_agent.py
```

`requirements.txt` carries the same `gnureadline` macOS-only marker as
the inline block.

> Invoke with `python3 sys_agent.py`, not `./sys_agent.py` — the shebang
> is `#!/usr/bin/env -S uv run --script`, which routes execution through
> uv and bypasses the venv you just activated.
>
> `pip install -U pip` is not optional on a freshly released Python. A
> stale pip doesn't recognize the new interpreter's wheel tag, falls back
> to building `pydantic-core` from source, and fails against PyO3's
> supported-version ceiling.

## Install a terminal shortcut

The getting-started instructions run the script from its downloaded folder. To
make `sys_agent` available elsewhere, run these commands from that folder:

```bash
chmod +x sys_agent.py
mkdir -p ~/.local/bin
ln -sf "$PWD/sys_agent.py" ~/.local/bin/sys_agent
```

Your shell must include `~/.local/bin` in `PATH` (its list of folders to search
for commands). If `sys_agent` is not found, continue using
`uv run --python 3.13 sys_agent.py` from the project folder, or add
`~/.local/bin` to your shell configuration.

## Python and terminal compatibility

Python 3.10–3.13 is the recommended range for this project. The quick start
selects 3.13 to avoid dependency build problems on newer interpreters: on a
just-released Python, pip may fall back to a Rust source build of
`pydantic-core` (pulled in by both SDKs) that fails against PyO3's
supported-version ceiling.

On macOS, installation includes `gnureadline` for line editing and colored
prompts; Linux uses the stdlib GNU readline. On Windows the stdlib lacks
`readline` entirely — the script still runs, but loses history persistence,
Up/Down recall, and line-editing keystrokes.
