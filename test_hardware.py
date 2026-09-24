"""Host-behaviour verification for sys_agent. Offline (no API key), but needs a
pty and a real process table, so it is NOT part of the standard baseline —
run it on each target host before tagging.

Covers what test_consult_render.py structurally cannot: signal delivery and
process-group teardown, the approval prompt under real readline, and env-file
resolution at real startup. These are the paths that differ between the Pi
(stdlib GNU readline, Debian) and the Mac (gnureadline or libedit), so a pass
here is per-host evidence, not a one-time result.

The REPL checks drive the real run_repl() under a pty with a scripted fake
provider, so readline, prompt_approval() and the tool loop all run for real.

NO-SPAWN INVARIANT: execute() is the single point where an approved command
becomes a subprocess, and every REPL check replaces it with a recorder. No
scripted turn can run anything — including a run with a guard deliberately
mutated out, where the point is precisely that the command WOULD otherwise
have run. A mutation check must neutralise the payload in the same step that
removes the guard; removing the guard is what makes the payload live. Do not
restore the real execute() in a REPL check.

Run: python3 test_hardware.py
"""
from __future__ import annotations

import json
import os
import pty
import select
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sys_agent as S                                        # noqa: E402

fails: list[str] = []


def report(tag: str, ok: bool, detail: str = "") -> None:
    print(f"[{tag}] {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        fails.append(tag)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _wait_for_pid(path: str, timeout: float = 6.0) -> int:
    """Read a pid a spawned shell wrote to `path`, or -1."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.1)
        try:
            txt = open(path).read().strip()
            if txt:
                return int(txt)
        except (OSError, ValueError):
            pass
    return -1


# -----------------------------------------------------------------------------
# Signals and process groups
# -----------------------------------------------------------------------------

def check_ctrl_c() -> None:
    """Real SIGINT through execute(). A driver process runs a command with a
    backgrounded descendant; we signal it the way a terminal delivers Ctrl-C
    and confirm the whole group is gone rather than orphaned."""
    pidfile = tempfile.NamedTemporaryFile("r", suffix=".pid", delete=False)
    pidfile.close()
    driver = (
        "import sys\n"
        f"sys.path.insert(0, {HERE!r})\n"
        "import sys_agent as S\n"
        "try:\n"
        f"    S.execute('sleep 300 & echo $! > {pidfile.name}; wait')\n"
        "except KeyboardInterrupt:\n"
        "    pass\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", driver],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    child = _wait_for_pid(pidfile.name)
    spawned_ok = child > 0 and alive(child)
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    time.sleep(1.0)
    ok = spawned_ok and not alive(child)
    if child > 0 and alive(child):
        os.kill(child, signal.SIGKILL)
    report("ctrl-c", ok, f"(descendant {child} reaped: {not alive(child)})")


def check_timeout() -> None:
    """Timeout path: exit 124 and no surviving descendant."""
    pidfile = tempfile.NamedTemporaryFile("r", suffix=".pid", delete=False)
    pidfile.close()
    saved = S.COMMAND_TIMEOUT
    S.COMMAND_TIMEOUT = 3
    start = time.monotonic()
    result = S.execute(f"sleep 120 & echo $! > {pidfile.name}; wait")
    elapsed = time.monotonic() - start
    S.COMMAND_TIMEOUT = saved
    child = _wait_for_pid(pidfile.name, timeout=1.0)
    time.sleep(0.8)
    ok = (result.returncode == 124 and "TIMEOUT" in result.stderr
          and child > 0 and not alive(child))
    if child > 0 and alive(child):
        os.kill(child, signal.SIGKILL)
    report("timeout", ok, f"(rc={result.returncode} in {elapsed:.1f}s, "
                          f"descendant reaped: {not alive(child)})")


def check_sigterm_ignorer() -> None:
    """The case the v1.17.0 fix targets: a descendant that ignores SIGTERM must
    still be SIGKILLed. Returning on the shell's exit (as _terminate_group once
    did) left it running, because the shell dies on the first SIGTERM."""
    pidfile = tempfile.NamedTemporaryFile("r", suffix=".pid", delete=False)
    pidfile.close()
    body = (
        'python3 -c "import signal,os,time;'
        'signal.signal(signal.SIGTERM, signal.SIG_IGN);'
        f"open('{pidfile.name}','w').write(str(os.getpid()));"
        'time.sleep(60)" & sleep 60'
    )
    proc = subprocess.Popen(body, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    child = _wait_for_pid(pidfile.name)
    started = child > 0 and alive(child)
    S._terminate_group(proc)
    time.sleep(0.5)
    ok = started and not alive(child)
    if child > 0 and alive(child):
        os.kill(child, signal.SIGKILL)
    report("sigterm-ignorer", ok, f"(escalated to SIGKILL: {not alive(child)})")


# -----------------------------------------------------------------------------
# REPL under a pty, driven by a scripted provider
# -----------------------------------------------------------------------------

# Markers, not str.format/f-strings: the template contains both dict literals
# and backslash escapes, and nesting either inside a format string is how this
# harness broke twice during development.
_DRIVER = '''
import json, os, sys
sys.path.insert(0, "@HERE@")
os.environ["SYS_AUDIT_LOG"] = "@AUDIT@"
os.environ["SYS_DISCLAIMER"] = "off"
import sys_agent as S

S.init_color()
S.init_readline()
S.init_audit()

# NO-SPAWN INVARIANT -- see this module's docstring. Never restore the real
# execute() here: it is what bounds every REPL check to zero execution, and a
# mutation run deliberately removes the guard that would otherwise stop a
# destructive command from reaching this point.
def _record(cmd):
    with open("@SPAWN@", "a") as fh:
        fh.write(cmd + chr(10))
    return S.CmdResult(cmd, 0, "(execution stubbed by test_hardware)", "")

S.execute = _record

SCRIPT = json.loads("""@SCRIPT@""")
PROVIDER = object.__new__(S.OpenAIProvider)
PROVIDER.name, PROVIDER.model = "openai", "gpt-5.4-mini"
_calls = [0]

def chat(messages, system, thinking=False, effort="high"):
    i = _calls[0]
    _calls[0] += 1
    # Echo what the model received, JSON-escaped so embedded newlines survive
    # the pty and a check can assert on exact message boundaries.
    print("@USER@" + json.dumps(messages[-1].get("content")), flush=True)
    spec = SCRIPT[i] if i < len(SCRIPT) else {"text": "done", "calls": []}
    calls = [S.ToolCall(c["id"], c.get("name", "run_command"), c["args"])
             for c in spec.get("calls", [])]
    return S.ChatTurn(text=spec.get("text", ""), tool_calls=calls,
                      raw_message={"role": "assistant",
                                   "content": spec.get("text", "")},
                      usage=S.Usage(input_tokens=1, output_tokens=1))

PROVIDER.chat = chat
S.run_repl(PROVIDER)
print("REPL-EXITED-CLEANLY")
'''


def run_repl_pty(script_turns: list[dict],
                 steps: list[tuple[str, str]],
                 timeout: float = 60.0) -> tuple[str, str]:
    """Drive run_repl() on a pty. `steps` are (expect_substring, keys) pairs
    sent in order, each only once its substring has appeared — blind timed
    writes race the prompt. Returns (terminal_output, commands_recorded)."""
    workdir = tempfile.mkdtemp()
    spawn_log = os.path.join(workdir, "spawned")
    driver_path = os.path.join(workdir, "drv.py")
    source = (_DRIVER
              .replace("@HERE@", HERE)
              .replace("@AUDIT@", os.path.join(workdir, "audit.log"))
              .replace("@SPAWN@", spawn_log)
              .replace("@SCRIPT@", json.dumps(script_turns)))
    with open(driver_path, "w") as fh:
        fh.write(source)

    pid, fd = pty.fork()
    if pid == 0:                                  # child: becomes the REPL
        os.execv(sys.executable, [sys.executable, driver_path])
    out = b""
    pending = list(steps)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.4)
        if ready:
            try:
                chunk = os.read(fd, 65536)
            except OSError:                       # pty closed on child exit
                break
            if not chunk:
                break
            out += chunk
        if pending:
            want, keys = pending[0]
            if want.encode() in out:
                time.sleep(0.4)                   # let the prompt finish drawing
                try:
                    os.write(fd, keys.encode())
                except OSError:
                    break
                out += b"\x00<sent>"              # consume this expectation once
                pending.pop(0)
        elif b"REPL-EXITED-CLEANLY" in out:
            break
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    recorded = open(spawn_log).read() if os.path.exists(spawn_log) else ""
    return out.decode(errors="replace"), recorded


def check_deny_edit() -> None:
    """`e` at the approval prompt, edited into a denied command. The deny check
    in the tool loop ran against the model's proposal, so the edited string is
    what must be re-checked before it reaches execute()."""
    turns = [{"text": "", "calls": [{"id": "c1", "args": {
        "command": "df -h", "explanation": "check disk"}}]}]
    out, recorded = run_repl_pty(turns, [
        ("you@", "check disk\n"),
        ("Run? [y]es", "e\n"),
        ("edit>", "\x15rm -rf /\n"),              # C-u clears the prefill
        ("you@", "/exit\n"),
    ])
    # The gate held only if the denied string never reached the spawn point.
    ok = ("[blocked]" in out and "REPL-EXITED-CLEANLY" in out
          and "df -h" in out and "api error" not in out
          and "rm -rf" not in recorded)
    report("deny-edit", ok,
           f"(blocked before execute; recorded={recorded.strip()!r})")


def check_tool_args() -> None:
    """Malformed tool arguments must not end the session. The tool loop sits
    outside the turn's try/except, so an AttributeError here used to escape
    run_repl() and drop the conversation."""
    turns = [
        {"text": "", "calls": [{"id": "d1", "args": ["not", "a", "dict"]}]},
        {"text": "", "calls": [{"id": "d2", "args": {"command": 5}}]},
    ]
    out, _ = run_repl_pty(turns, [("you@", "probe\n"), ("you@", "/exit\n")])
    ok = ("REPL-EXITED-CLEANLY" in out and "Traceback" not in out
          and "AttributeError" not in out and "api error" not in out)
    report("toolargs", ok, "(non-dict args + non-string command survived)")


def check_paste_block() -> None:
    """A \"\"\"-delimited block must reach the model as ONE message. The block
    is written to the pty in a single burst, as a terminal paste arrives, so
    the continuation input() calls must drain the rest of the lines from the
    tty buffer under the live readline backend. Also: Ctrl-C discards a
    partial block without sending it or ending the session, and a line inside
    the block that looks like a meta-command ("/exit") is content."""
    q = '"""'
    out, _ = run_repl_pty([], [
        ("you@", f"{q}\npartial draft\n"),
        ("... ", "\x03"),
        ("cancelled]", f"{q}\nline one\n  line two\n/exit\n{q}\n"),
        ('@USER@"line one', f"{q}hi there{q}\n"),
        ('@USER@"hi there"', "/exit\n"),
    ])
    sent = [json.loads(line.split("@USER@", 1)[1])
            for line in out.replace("\r", "").splitlines()
            if line.startswith("@USER@")]
    ok = (sent == ["line one\n  line two\n/exit", "hi there"]
          and "[multi-line input cancelled]" in out
          and "REPL-EXITED-CLEANLY" in out and "Traceback" not in out)
    report("paste-block", ok, f"(model received {sent!r})")


# -----------------------------------------------------------------------------
# Startup: env file resolution and readline backend
# -----------------------------------------------------------------------------

def check_config() -> None:
    """Env-file settings must apply at real startup, and a bad value must warn
    rather than abort. Both were captured at import before v1.17.0, so the
    documented file path silently did nothing."""
    workdir = tempfile.mkdtemp()
    env_path = os.path.join(workdir, ".env")
    with open(env_path, "w") as fh:
        fh.write("SYS_COMMAND_TIMEOUT=7\nSYS_TOP_PROCESSES=xyz\nSYS_COLOR=off\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY")}
    env["SYS_ENV_FILE"] = env_path
    proc = subprocess.run([sys.executable, os.path.join(HERE, "sys_agent.py")],
                          capture_output=True, text=True, env=env,
                          stdin=subprocess.DEVNULL)
    blob = proc.stdout + proc.stderr
    ok = ("loaded 3 vars" in blob            # file was read
          and "SYS_TOP_PROCESSES" in blob    # bad value reported, not fatal
          and "no API key set" in blob       # reached provider selection
          and "\x1b[" not in blob)           # SYS_COLOR=off from the file won
    report("config", ok, "(file applied, bad value warned, SYS_COLOR honoured)")

    bind = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {HERE!r})\n"
         "import sys_agent as S\n"
         f"S.load_env_file(S.find_env_file({env_path!r}))\n"
         "S.init_config()\n"
         "print('TIMEOUT', S.COMMAND_TIMEOUT)"],
        capture_output=True, text=True)
    report("config-bind", "TIMEOUT 7" in bind.stdout, f"({bind.stdout.strip()})")


def check_readline() -> None:
    """Which backend is live, and that a history path was bound. Pi-only
    readline bugs are real (the v1.4.1 history fix), so record the backend."""
    probe = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {HERE!r})\n"
         "import sys_agent as S\n"
         "S.init_readline()\n"
         "print('have', S._HAVE_READLINE, 'libedit', S._IS_LIBEDIT,\n"
         "      'hist', S._history_path)"],
        capture_output=True, text=True)
    ok = "have True" in probe.stdout and "hist " in probe.stdout
    report("readline", ok, f"({probe.stdout.strip()})")


def check_sudo_probe() -> None:
    """The startup `sudo` fact must predict what an approved sudo command
    actually gets. Runs a plain `sudo true` (the shape a model proposes, no
    -n) through the real execute(), outside any REPL, so the no-tty session
    is real. A harmless payload by construction; it cannot hang because
    execute() gives sudo no terminal to prompt on (it fails in ~0.1s), and
    the timeout bounds it regardless."""
    mode = S._sudo_mode()
    saved = S.COMMAND_TIMEOUT
    S.COMMAND_TIMEOUT = 10
    try:
        result = S.execute("sudo true")
    finally:
        S.COMMAND_TIMEOUT = saved
    ran = result.returncode == 0
    if mode == "passwordless" or mode == "running_as_root":
        ok = ran
    elif mode == "password_required":
        ok = not ran
    elif mode == "not_installed":
        ok = not ran and S.shutil.which("sudo") is None
    else:
        ok = False                      # inconclusive on a healthy host
    report("sudo-probe", ok,
           f"(fact={mode}, real sudo via execute() rc={result.returncode})")


def main() -> int:
    print(f"host: {os.uname().nodename} ({os.uname().sysname}/{os.uname().machine})")
    check_ctrl_c()
    check_timeout()
    check_sigterm_ignorer()
    check_deny_edit()
    check_tool_args()
    check_paste_block()
    check_config()
    check_readline()
    check_sudo_probe()
    print()
    print("RESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
