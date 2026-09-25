# sys_agent

Ask questions about your computer in plain language. sys_agent checks your
system, asks an AI service for the next step, and shows you each suggested
command for approval before running it.

Use it to investigate problems, check disk space, understand running services,
and maintain a Mac, Linux computer, or Raspberry Pi.

<!-- BEGIN EXAMPLE -->
![Example sys_agent session showing a question and a proposed command](assets/example-session.svg)
<!-- END EXAMPLE -->

[Why sys_agent](#why-sys_agent) · [Get started](#get-started) ·
[Your first session](#your-first-session) ·
[Everyday commands](#everyday-commands) · [More help](#more-help)

## Why sys_agent

AI coding tools are built for writing code. sys_agent is built for a different
job: everyday system administration, where the right answer depends on the
machine in front of you — which operating system it runs, how it starts and
stops background services, how it installs software, and which tools are
actually present.

So it looks at the host before it asks anything. Hardware model, OS, disks,
running services, and available tools are collected at startup and sent with
every question, so the commands that come back already fit this machine — the
same question gets a different answer on a Raspberry Pi than on a cloud server,
without a round of guessing first. See
[collected host information](docs/technical-reference.md) for the full list.

## Before you start

You will need:

- **A Mac or Linux computer**, including Raspberry Pi, with internet access.
- **A terminal** in which to paste the setup commands below.
- **Git and uv** installed. Git downloads this project; [uv](https://docs.astral.sh/uv/)
  manages Python and the libraries it needs.
- **An API key from OpenAI, Anthropic, or DeepSeek.** This is a private access
  key for the AI service you choose. One provider is enough. Check that
  provider's API billing before use; do not assume a chat subscription
  includes it. OpenAI's GPT-6 Astra does not work with sys_agent yet; GPT-6
  Luna and Sol, and the default OpenAI model, do. See
  [why](docs/advanced-usage.md#unsupported-openai-gpt-6-astra).

### What runs locally, and what is shared?

Commands run on the computer where you start sys_agent, with your user
account's permissions. There is no sandbox separating them from your files.

Your questions, collected system information, and command results are sent to
the selected AI provider. System information includes hardware, disks,
running services, and whether sudo works without a password. Command output
can contain sensitive information, so consider what a proposed command will
read before approving it. The assistant is told not to send information from
your computer to other websites unless you ask, but check any command that
contacts the internet before approving it.

### You control execution

Approval is required by default. Read each command and its explanation before
running it: AI suggestions can be wrong. **Pressing Enter approves the
command.**

A short deny list blocks some destructive commands, but it cannot recognize
every dangerous operation. Stopping a session does **not** undo changes already
made. Automatic approval ([`/auto on`](docs/advanced-usage.md#meta-commands))
runs every command without asking; leave it off while getting familiar with the
tool.

## Get started

These steps use macOS or Linux. Paste shell commands into your terminal;
questions and slash commands go into sys_agent after it starts.

### 1. Check the prerequisites

```bash
git --version
uv --version
```

Both should print a version number. If either command is missing, install it
before continuing. See the [Git installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
and [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Download sys_agent

```bash
git clone https://github.com/mikeoc61/sys_agent.git
cd sys_agent
```

Keep this terminal open in the `sys_agent` folder for the remaining steps.

### 3. Add your API key

Create a private configuration file:

```bash
mkdir -p ~/.config/sys_agent
touch ~/.config/sys_agent/.env
chmod 600 ~/.config/sys_agent/.env
nano ~/.config/sys_agent/.env
```

In the editor, add **one** of these lines for the provider you use. Replace
`YOUR_API_KEY` with your actual key. Do not paste all three unless you have
keys for all three providers.

```text
OPENAI_API_KEY=YOUR_API_KEY
ANTHROPIC_API_KEY=YOUR_API_KEY
DEEPSEEK_API_KEY=YOUR_API_KEY
```

In nano, press **Ctrl-O**, then **Enter** to save, and **Ctrl-X** to exit.
If nano is unavailable, open the same file in a text editor of your choice.
Keep this file private; do not share your key in screenshots or messages.

These steps use the default configuration location. If you have customized
`XDG_CONFIG_HOME`, use that folder instead of `~/.config`.

Two things take precedence over this file: variables already exported in your
shell, and a `.env` file in the folder you launch from. The project folder from
step 2 is searched *first*, so if a `.env` ever appears there it wins over the
one you just created — a surprise worth knowing about before you go looking
for a key that seems to be ignored. The project also ships `.env.example`, a
commented template covering the common settings; see
[configuration lookup](docs/configuration.md#configuration) for both, and
[all environment variables](docs/configuration.md#all-environment-variables)
for the complete list.

### 4. Start it

From the downloaded `sys_agent` folder:

```bash
uv run --python 3.13 sys_agent.py
```

On the first run, uv downloads Python if needed and installs the required
libraries in its cache. Later runs reuse them. If you configured more than one
AI provider, sys_agent asks which one to use.

Use the same command from this folder for future sessions. You can also set up
an optional [terminal shortcut](docs/configuration.md#install-a-terminal-shortcut)
to launch it as `sys_agent` from other folders.

When a newer version is available on GitHub, sys_agent says so when it starts
and shows the command that updates it, which works from any folder. From the
`sys_agent` folder, that command is:

```bash
git pull
```

`/version` shows which version you are running at any time. The check only asks GitHub which version is current; none of your system
information is sent. See
[Update check](docs/technical-reference.md#update-check) for details.

## Your first session

Start with a question that only asks for information:

```text
How much disk space do I have left?
```

Other questions to try:

- "Which programs are using the most memory?"
- "How long has this computer been running?"
- "Help me understand why this service keeps restarting."

The prompt shows the AI model and the computer you are working on. When the AI
suggests a command, sys_agent shows the command, its reason, and the folder
where it will run (`CWD`).

### Approve, skip, or stop

| Your input | What happens |
|---|---|
| `y` or Enter | Run the displayed command. |
| `n` | Skip this command. The AI may suggest another step. |
| `e` | Edit the command, then run the edited version if the deny list allows it. |
| `q` | Stop the current task and return to the question prompt. |

After an approved command finishes, its output goes back to the AI. It may
explain the result or propose another command for approval. You can ask
follow-up questions in the same session.

**To interrupt:** press Ctrl-C during a task. sys_agent cancels the current
operation and returns to the question prompt. Commands that already completed
are not undone. The interrupted exchange is removed from the conversation,
while its command audit records remain. With automatic approval on, Ctrl-C is
the only way to stop a run, because there is no per-command prompt to answer.

**To exit:** type `/exit` at the question prompt, or press Ctrl-D there.
Ctrl-C at an empty prompt clears the line without quitting.

## Everyday commands

Type these inside sys_agent:

| Command | Use it to… |
|---|---|
| `/help` | See all available commands. |
| `/history` | Review recent commands and what happened to them. |
| `/reset` | Start a fresh conversation and clear token counters. |
| `/info` | See the selected AI model and collected system information. |
| `/version` | See which version you are running and whether it is up to date. |
| `/facts refresh` | Collect fresh system information and start a new conversation. |
| `/consult` | Get a second opinion from your other configured AI providers. |
| `/exit` | End the session. |

Use **Up/Down** to recall previous questions and **Ctrl-R** to search them.
Question history is saved between sessions; the AI conversation itself is not.
Other [line-editing keys](docs/advanced-usage.md#line-editing-and-history)
work as they do in a normal shell.

### Paste several lines at once

Normally, pressing Enter sends your question, so pasted text with several lines
would be sent one line at a time. To send it as one message, type `"""` and
press Enter, paste your text, then type `"""` and press Enter again:

```text
"""
Why does this error keep appearing?
kernel: EXT4-fs warning: mounting fs with errors, running e2fsck is recommended
"""
```

Press Ctrl-C before the closing `"""` to discard the text. See
[multi-line input](docs/advanced-usage.md#multi-line-input) for details.

### Get a second opinion

If you have keys for more than one provider, `/consult` asks the others how
they would approach your latest question. It displays suggestions without
running any commands and makes additional API requests.

To compare suggestions before executing anything, stop at the first approval
prompt with `q`, then type `/consult`. See
[second opinions](docs/advanced-usage.md#multi-provider-consult) for details
and the option to exclude earlier conversation history.

### Review what happened

`/history` shows a readable command log, including skipped and blocked commands.
If it opens in a pager, press `q` to close that view. The log is enabled by
default; command output is excluded unless you explicitly enable its capture.

Your API-key file, saved question history, and command log normally live under
`~/.config/sys_agent/`. See [configuration files](docs/configuration.md#files)
for their names and settings.

## More help

- **[Configuration reference](docs/configuration.md):** all settings, alternative
  installation, file locations, and the optional terminal shortcut.
- **[Advanced usage](docs/advanced-usage.md):** all slash commands, choosing models,
  second opinions, thinking settings, and command history.
- **[Technical reference](docs/technical-reference.md):** collected host information,
  disk-health detection, execution controls, audit format, and architecture.

sys_agent is designed for interactive system administration. It is not a coding
agent — use a tool built for authoring code for that — and it is not
sandboxed: commands run as you, with your full privileges. It does not run
scheduled jobs or retain an AI conversation between sessions. Windows has
reduced terminal functionality; macOS and Linux are the primary targets.

## License

MIT — see [LICENSE](LICENSE).
