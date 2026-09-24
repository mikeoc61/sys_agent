# Technical reference

[Getting started](../README.md) · [Configuration](configuration.md) ·
[Advanced usage](advanced-usage.md) ·
[Technical reference](technical-reference.md)

Implementation details for maintainers and users who want to inspect how host
information, execution, and logging work.

## Runtime snapshot: services and processes

Alongside the static host facts, startup probes capture a point-in-time
snapshot of what the machine is actually running, so the model uses real
service and process names on the first turn instead of burning a round trip to
discover them. Four keys are injected into the facts when non-empty:

- **`running_services`** — the active services. On Linux, the running `systemd`
  service units (`systemctl list-units --type=service --state=running`); empty
  on hosts without `systemd`, or where the bus is unreachable (e.g. inside a
  container). On macOS, the currently-running `launchctl` jobs with Apple
  system agents and per-GUI-app jobs filtered out (see the platform note),
  leaving the user-relevant daemons (Homebrew, vendor helpers, custom
  LaunchAgents/Daemons).
- **`running_services_user`** (Linux/systemd) — the running **per-user** units
  from `systemctl --user`, the manager where a user's own services live (e.g. a
  `bitcoind.service` you start with `systemctl --user`, invisible to the
  system-scope list above). Present only when non-empty, which requires a
  reachable user D-Bus session: present when sys_agent runs in the invoking
  user's login session (keep it alive headless with
  `loginctl enable-linger <user>`), absent under sudo, cron, or a user without
  linger. The system prompt tells the model to match scope to the list a unit
  appears in — `systemctl --user status <unit>` and `journalctl --user -u
  <unit>` for these (no sudo), `systemctl status` / `journalctl -u` for the
  system list — and that `journalctl` is the log source for a service rather
  than hunting `/var/log`.
- **`failed_services`** (Linux/systemd) — SYSTEM service units in a **failed**
  state at probe time (`systemctl list-units --type=service --state=failed`),
  empty on a healthy host and under the same conditions that empty
  `running_services` (no systemd, unreachable bus). High-signal for turn-one
  triage: a unit named here is a failure the model would otherwise spend a
  discovery pass finding. The system prompt has it confirm the unit is still
  failed and read the cause (`systemctl status <unit>`, `journalctl -u <unit>`)
  before acting, since a unit may have been restarted since startup. SYSTEM
  scope only — user-scope failures carry the same bus-availability caveat as
  `running_services_user` and are left to an explicit `systemctl --user
  --failed`.
- **`top_processes`** — the top processes by resident memory (RSS), 10 by
  default (`SYS_TOP_PROCESSES`). Processes sharing a name are aggregated into a
  single entry with summed RSS and an instance `count`, so a worker pool reads
  as one `gunicorn ×8` line rather than eight rows. On Linux this is read
  straight from `/proc` — no `ps` dependency, so it behaves identically across
  distros, init systems, and userlands; on macOS via `ps -axo rss=,comm=`.

Both are a **snapshot taken at startup**, not live state: a process that is hot
at launch may be idle later, and the service list reflects startup. The model
is instructed to use them for naming and orientation but to confirm current
state with a command before acting on a reading. `/facts refresh` re-probes.

**Privacy.** Only the process *name* is recorded — the executable basename,
never its arguments — so secrets passed on a command line (`--password=…`, API
tokens) are dropped before anything is sent to the model.

**Platform note.** The two service lists are intentionally asymmetric. Linux
includes system units (`dbus`, `polkit`, `systemd-*`); macOS drops the
equivalent Apple tier (`com.apple.*`) plus launchd's per-GUI-app jobs
(`application.*`), because launchd registers hundreds of these that would
otherwise bury the handful of services worth naming. The macOS filter lives in
the `_DARWIN_SERVICE_SKIP_PREFIXES` tuple in `sys_agent.py`; extend it if a
host's noise differs. Expect the service list to carry far more signal on a
server — where it names your actual workload — than on a desktop, where it is
mostly background updaters.

## Hardware identity

Alongside the OS facts, startup resolves what the machine *is* — not just
which OS it runs — and injects it as a `hardware` block:

```json
"hardware": {
  "model": "Raspberry Pi 5 Model B Rev 1.0",
  "platform_class": "sbc",
  "memory_gb": 7.9,
  "swap_gb": 0.2
}
```

On a recognized cloud instance the block also carries a `cloud` label:

```json
"hardware": {
  "model": "t3.micro",
  "vendor": "Amazon EC2",
  "platform_class": "vm",
  "memory_gb": 0.9,
  "swap_gb": 0.0,
  "cloud": "aws"
}
```

| Field | Source (Linux) | Source (macOS) |
|---|---|---|
| `model` | `/proc/device-tree/model` (SBCs), else DMI `product_name` | `sysctl hw.model` |
| `vendor` | DMI `sys_vendor` | `Apple` |
| `platform_class` | see below | battery in `pmset -g batt` → `laptop`, else `desktop` |
| `memory_gb` | `/proc/meminfo` MemTotal | `sysctl hw.memsize` |
| `swap_gb` | `/proc/meminfo` SwapTotal (`0` = none) | — |
| `cloud` | DMI vendor/product needle, else `bios_version`/`bios_vendor` (Xen-gen EC2), else Azure `chassis_asset_tag` | — |
| `cpu_model` | — | `machdep.cpu.brand_string` |

`platform_class` is one of `container | vm | sbc | laptop | desktop | server`,
resolved in that precedence order (a Pi inside Docker reads as `container`):
container detection (`/.dockerenv`, cgroup probe), then
`systemd-detect-virt --vm` (DMI vendor/product needles as fallback), then
device-tree presence, then the SMBIOS chassis-type code. Placeholder DMI junk
("To Be Filled By O.E.M.") is filtered out rather than surfaced as identity.
Fields are attached only when resolved; everything is static, unprivileged,
and effectively free at startup. On macOS the laptop/desktop split keys on
battery presence, not the model string — Apple Silicon identifiers
(`Mac15,6`) no longer encode the form factor.

**Why it matters.** The hardware identity tells the model which telemetry
classes are even plausible before it proposes anything: PMIC rails and
`vcgencmd` on a Raspberry Pi, battery/`pmset`/`upower` on a laptop, IPMI on
a server, "you can't measure that" inside a VM. The system prompt pairs the
block with two rules: a negative capability claim ("there is no live wattage
reading on this host") must be verified with a one-line probe before being
asserted, and on a Raspberry Pi the model is pointed directly at
`vcgencmd get_throttled` and (Pi 5) `vcgencmd pmic_read_adc` for live
undervoltage/throttle state and per-rail power draw. The startup tool probe
now also detects `vcgencmd`, `sensors`, `upower`, `dmidecode`, and
`ipmitool`, so tool presence corroborates the hardware identity.

The default facts also enumerate `/sys/class/hwmon/*/name` on Linux as
`hwmon_sensors` (e.g. `cpu_thermal`, `rpi_volt`, `pwmfan`, `nvme`) — derived
from sysfs, not a curated map — so the sensor channels that exist on the
host are listed in context rather than guessed.

**Cloud classification.** When DMI identifies a recognized cloud platform, the
hardware block carries a `cloud` label (`aws | gcp | azure | oci |
digitalocean`). It is a *classification*, not identity — it says "an EC2
instance," never which instance or whose account — derived from the same
world-readable DMI fields VM detection already reads (vendor/product needles,
with Azure's fixed `chassis_asset_tag` as its discriminator, since Azure shares
the generic "Microsoft Corporation" vendor with on-prem Hyper-V). Nitro-
generation EC2 is caught by its `sys_vendor` of "Amazon EC2"; Xen-generation
instances (t2/m4/…) report `sys_vendor` "Xen" / `product_name` "HVM domU" with
no "amazon" string, so they fall back to the BIOS, which stamps
`bios_version` "4.11.amazon" (or `bios_vendor` "Amazon"). The world-readable
`/sys/hypervisor/uuid` would also identify Xen-EC2 but is a UUID-class file, so
it is deliberately left unread. No metadata service is contacted at startup.

The label steers command strategy: on a cloud VM, storage is network-attached
(a full root disk is a volume resize via the provider, not local `parted`) and
connectivity is governed by provider-level security groups / NACLs on top of
host `iptables`/`ufw`, so a capacity or reachability question may have its real
answer at the provider. Instance metadata (region, AZ, instance-type,
instance-id, IAM role) is deliberately **not** pre-loaded; the system prompt
has the model query IMDS (`169.254.169.254`, IMDSv2 token flow) or GCP's
`metadata.google.internal` on demand only when a task needs it, and treat
instance-id, account-id, and IAM role as sensitive.

**Temporal orientation.** The default facts include `timezone`,
`boot_time` (ISO-8601 local), and `uptime_hours`, resolved from `/proc/uptime`
on Linux and `kern.boottime` on macOS. These are stable startup facts — the
model uses them to interpret log timestamps and gauge reboot recency without a
probe, while current wall-clock and live uptime are still a `date`/`uptime`
away. `swap_gb` rounds out the memory picture: a value of `0` means no swap is
configured, so on a low-RAM host memory pressure ends in OOM-kills rather than
swapping — context the model weighs when diagnosing killed processes.

## Sudo availability

Each approved command runs in a new session with no terminal, so sudo cannot
ask for a password there. A plain `sudo <cmd>` fails immediately with "a
terminal is required to read the password" unless the account has a
passwordless (`NOPASSWD`) rule. A password you typed into sudo in your own
terminal does not help, because sudo ties that cached credential to the
terminal it was entered on.

Startup probes this once with `sudo -n true`, spawned the same way as an
approved command, and injects the result as `sudo`:

| Value | Meaning |
|---|---|
| `passwordless` | sudo works from the agent |
| `password_required` | sudo fails from the agent, including when the user has no sudo rights |
| `running_as_root` | sys_agent itself runs as root; sudo is unnecessary |
| `not_installed` | no `sudo` on the host |

If the probe times out or cannot spawn, the key is omitted rather than
guessed. The system prompt tells the model not to propose sudo commands under
`password_required`. It should try an unprivileged command first. If root is
genuinely needed, it gives you the command as text to run yourself and asks
you to paste back the output. It must never ask for your password or work
around the prompt with `sudo -S` or an askpass helper.

Only running a command gives the right answer. Checking a cached credential
(`sudo -n -v`) fails on a passwordless Pi, and asking whether a command is
permitted (`sudo -n -l true`) succeeds on a Mac that needs a password. The
probe takes 5 to 50 ms. On a passwordless host it adds three lines to the
system auth log per startup or `/facts refresh`: the `/usr/bin/true` command
and a root session opening and closing. `test_hardware.py` checks the fact
against a real `sudo true` run through `execute()`.

## SMART over USB bridges

The injected host facts include a mount-first disk topology under `disks`
(mounts, sizes, fstype, and a `device` sub-record per physical disk with model,
transport, and rotational flag). On macOS, Time Machine snapshot automounts
under `/Volumes/.timemachine/` are filtered out — a host with regular local
backups mounts dozens of them, all reporting the destination volume's identical
totals — while the backup destination volumes themselves are kept. For
USB-attached disks, two extra fields are resolved at startup so the model can
read SMART/health without a probing round trip:

- **`usb_ids`** — the USB bridge's `vendor:product` (e.g. `04e8:4001`),
  resolved at runtime from `udevadm` (sysfs fallback when udevadm is absent).
- **`smartctl_device_type`** — the `smartctl -d` pass-through token for that
  bridge (e.g. `sntasmedia`), present only when the bridge is recognized.

```json
"disks": {
  "mounts": [{
    "mount": "/media/mikeoc/T72GB",
    "device": {
      "name": "/dev/sda",
      "model": "PSSD T7",
      "transport": "usb",
      "rotational": false,
      "usb_ids": { "vendor": "04e8", "product": "4001" },
      "smartctl_device_type": "sntasmedia"
    }
  }]
}
```

**Why this is needed.** A USB enclosure hides whether the drive behind it is
SATA or NVMe, so a bare `smartctl -a /dev/sdX` often fails to auto-detect it and
the correct `-d` token is bridge-chip-specific. With `smartctl_device_type`
present the model goes straight to `smartctl -d sntasmedia -a /dev/sda`; when
only `usb_ids` is present (an unrecognized bridge) the prompt steers it to try
`-d sat`, then the NVMe pass-through types.

**Minimal hard-coding.** The `vendor:product` pair is derived at runtime — only
the small chip→token map (`_USB_BRIDGE_SMART_HINTS` in `sys_agent.py`) is
hard-coded, because that mapping is not derivable from anything the kernel
exposes. It mirrors the `KNOWN_BRIDGE_HINTS` table in the external companion
`disk_smart.py` (not included in this repository); maintainers using both
should keep their mappings in sync.

**Interpreting NVMe counters.** The system prompt also tells the model that
Unsafe Shutdowns, Warning/Critical Composite Temperature Time, and Error Log
Entries are *cumulative lifetime totals* — to be read as a rate of change, not
as alarming absolutes — and that an unsafe-shutdown count is low-signal on a
USB-bridged drive (many bridges never forward the NVMe shutdown notification, so
it climbs even on clean unmounts).

## Update check

At startup a background thread checks whether the running `sys_agent.py` is
the version on GitHub `main`. It reports only; it never downloads or replaces
code, because an agent that runs shell commands must not change itself
without being asked.

1. The git blob hash of the running file (what `git hash-object` prints) is
   compared with the `sha` that the GitHub contents API reports for
   `sys_agent.py` on `main`. If they match, nothing is printed. No version
   constant is involved, so the check works for a clone, a copied file, or
   `uv run`.
2. If they differ and the file is in a git checkout, `git rev-parse HEAD` gives
   the local commit, and the compare API (`compare/<HEAD>...main`) classifies
   it. GitHub answers, so the check never runs `git fetch` or touches the
   repository.

| Situation | Startup line |
|---|---|
| File matches `main` | none |
| Checkout is behind `main` | `update available: N commits behind GitHub main — run: git -C <dir> pull` |
| Checkout has diverged from `main` | `this checkout has diverged from GitHub main (N behind, M ahead)` |
| HEAD not on GitHub (unpushed commits) | `… differs from GitHub main and this checkout has commits GitHub does not` |
| Not a git checkout, or git missing | `sys_agent.py differs from GitHub main (an update, or local edits)` |
| Checkout ahead of `main`, or equal with uncommitted edits | none |

The compare direction is easy to misread: `compare/BASE...HEAD` describes
`main` relative to the local commit, so status `ahead` means the local copy is
behind.

Startup waits at most one second for the result at the banner. A slower result
is printed before a later prompt, never over the input line. Any failure
(offline, DNS, the unauthenticated 60 requests/hour limit, unexpected JSON) is
silent. Each startup makes at most two requests to `api.github.com`. They
carry your IP address, a `sys_agent-update-check` User-Agent, and, in the
second request, the local HEAD commit id. No host facts are sent. Disable with
`SYS_UPDATE_CHECK=off`.

[`/version`](advanced-usage.md#version-and-update-status) runs the same check
on demand and prints every outcome, including the ones startup keeps quiet
(up to date, ahead, local edits, and the reason a check failed). It runs even
with `SYS_UPDATE_CHECK=off`, because the user asked for it.

## Safety model

Execution controls:

1. **Approval prompt** (default-on, per-command). Every `run_command` shows
   the exact shell string, the model's stated reason, and the CWD before any
   subprocess is spawned. Default answer is `y` so casual `Enter` runs it —
   read the line. Per command: `y` run, `n` skip (the agent
   continues), `e` edit, `q` stop the whole workflow. Ctrl-C stops the
   workflow from anywhere in the turn — killing a running command's process
   group first — and returns to the prompt; it never ends the session. Under
   `/auto on` there is no per-command prompt, so Ctrl-C is the only way to
   break out of a run.
2. **Local hard-deny list** (always-on). A short set of irrecoverable command
   patterns is blocked before the approval prompt is even shown. The model
   cannot disable this and `/auto on` cannot bypass it. Matching is
   intent-based (argv inspection through wrappers like `sudo`/`env`/`timeout`).
   Editing a command with `e` re-runs the check on the edited string, so what
   is actually spawned is always what was matched — an edit cannot walk a
   denied pattern past the gate. See `is_denied()` and the `_DENY_*` /
   `_FORKBOMB_RE` tables in `sys_agent.py`.
3. **Command timeout** (120s wall-clock per command, configurable via
   `SYS_COMMAND_TIMEOUT`). Prevents runaway model loops from hanging the REPL
   on a single command. On timeout — and on Ctrl-C — the command's entire
   process group is signalled, not just the shell: SIGTERM first, then SIGKILL
   for anything still in the group after a grace period. A descendant that
   ignores SIGTERM (or outlives the shell that spawned it) is still cleaned
   up while it remains in that process group. Descendants that create a separate
   session or group are outside this cleanup mechanism.

Ahead of all three sits a **stop-reason guard** in the REPL loop. A reply
that the API cut short (`stop_reason` `max_tokens` /
`model_context_window_exceeded`, `finish_reason` `length`) or declined
(`refusal`, `content_filter`) arrives as HTTP 200 with well-formed-looking
content, and a `tool_use` input truncated mid-string parses as a valid,
shorter command — one the deny list and the approval prompt would both judge
on its face. `chat()` classifies the stop into `ChatTurn.stop`
(`refusal` / `truncated`) on every provider, and the loop discards a stopped
turn's tool calls before the deny check or prompt ever sees them, printing
what happened and why (see `docs/advanced-usage.md`, "Declined and truncated
replies"). A refused turn is rolled back like Ctrl-C; a truncated tool turn
is dropped like an API error, so no dangling `tool_use` is left awaiting a
`tool_result`. Guarded by `test_consult_render.py` (fake clients through the
real `chat()`, a scripted `run_repl()` with `execute()` stubbed), with the
mutation check that removing the guard sends the cut command to the prompt.

A dimmed startup notice restates the premise of layer 1: model-proposed
commands can be confidently wrong (hallucinated flags, paths, unit names;
stale syntax), and the approval prompt is where you catch that. Suppress it
with `SYS_DISCLAIMER=off`.

Two system-prompt rules cover what the model proposes rather than how it
runs. Neither is enforced in code; the approval prompt is where a violation is
caught.

- **Sudo.** The rule keys on the `sudo` fact described in
  [Sudo availability](#sudo-availability).
- **Host data stays on the host.** The model must not send anything read from
  the machine, such as MAC or IP addresses, serial numbers, hostnames, file
  contents, logs or keys, to an external service unless you asked for that
  transfer. A MAC vendor lookup is the standard case. It resolves offline
  on systemd hosts with `systemd-hwdb query OUI:XXXXXX`, and from nmap's or
  `ieee-data`'s OUI files where installed. When no local source exists, the
  model says so and leaves the choice to you. Requests that carry nothing
  from the host, such as package updates, are unaffected.

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
| `edited_command` | `action=edit`, or `action=deny` for a blocked edit | the command as you rewrote it before running |
| `returncode` | run/edit | process exit code (`124` = timeout) |
| `truncated` | run/edit | whether output to the model was clipped at `OUTPUT_MAX_CHARS` |
| `reason` | `action=deny` | which hard-deny rule matched (on the proposed command, or on your edit) |
| `note` | as needed | e.g. `interrupted (workflow stopped)` |
| `stdout` / `stderr` | only with `SYS_AUDIT_BODY=on` | command output, capped at `OUTPUT_MAX_CHARS` |

`abort` records a workflow stopped at the approval prompt (`q` / Ctrl-C)
before that command ran. A command interrupted *mid-execution* by Ctrl-C is
logged as `run`/`edit` with `note: interrupted (workflow stopped)` — it did
start, so its disposition is not `abort`.

Output **bodies are not logged by default** — stdout/stderr can carry secrets.
Enable with `SYS_AUDIT_BODY=on` only if you accept that.

Disable the log, send it elsewhere, or opt into output bodies:

```sh
SYS_AUDIT_LOG=off sys_agent
SYS_AUDIT_LOG=/var/log/sys_agent.jsonl sys_agent
SYS_AUDIT_BODY=on sys_agent
```

At runtime: `/audit` shows status; `/audit on|off` toggles. Disposition follows
provider/model/host live, so a mid-session `/provider`, `/model`, or `/facts`
switch is reflected in subsequent records. The log is not rotated (one short
line per command); truncate with `> ~/.config/sys_agent/audit.log`.

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
                  ┌─────────────────────────────────────────────┐
                  │            Provider abstraction             │
                  │ OpenAIProvider | AnthropicProvider |        │
                  │ DeepSeekProvider (OpenAI-compatible)        │
                  └──────────────┬──────────────────────────────┘
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
                                 │    │ subprocess.Popen(...)
                                 │    │
                                 ▼    ▼
                  ┌────────────────────────────────────┐
                  │         Local host                 │
                  └────────────────────────────────────┘
```

The provider abstraction normalizes tool-call/tool-result message structure
across the APIs (OpenAI and DeepSeek send one `role: tool` message per call;
Anthropic batches all `tool_result` blocks into a single user message).
`DeepSeekProvider` subclasses `OpenAIProvider` — same wire shape over DeepSeek's
OpenAI-compatible endpoint — overriding only client construction (base URL +
key) and the thinking path (`reasoning_effort` plus `reasoning_content`
preservation). The REPL is provider-agnostic.

### Packaging

Everything lives in one file. `sys_agent.py` declares its dependencies in a
PEP 723 inline metadata block, so `uv` builds and caches the environment on
first run with nothing installed system-wide; `requirements.txt` mirrors the
same set for the pip path. Dependencies are the provider SDKs plus
`gnureadline` on macOS — no agent framework.

### Runtime state

The REPL tracks active provider, active model, host facts, verbosity, token
counters, and conversation messages as mutable runtime state. `/provider`,
`/model`, and `/facts` update that state without restarting the process.
