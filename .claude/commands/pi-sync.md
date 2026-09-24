---
description: Push main, sync pibot's checkout, run offline checks there
---
1. Run `git status`. Stop if the tree is dirty or main is not ahead of origin/main.
2. Run the local baseline: `python3 -m py_compile sys_agent.py && python3 test_consult_render.py`. Stop on failure.
3. Run `git push origin main`.
4. Run `ssh -o BatchMode=yes pibot 'cd ~/projects/sys_agent && git pull --ff-only && git log -1 --oneline && python3 -m py_compile sys_agent.py && python3 test_consult_render.py'`.
5. Confirm pibot's HEAD matches local HEAD (`git rev-parse HEAD`). Report both hashes and the harness RESULT line.
6. If everything passed, run the hardware harness on pibot under both interpreters and report each RESULT line and `[readline]` backend: `ssh -o BatchMode=yes pibot 'cd ~/projects/sys_agent && ~/.local/bin/uv run --quiet test_hardware.py; python3 test_hardware.py'`.
7. Derive interactive checks from `git diff $(git describe --tags --abbrev=0)..HEAD` and run the ones that can be scripted on pibot yourself (stdlib pty driver under `uv run --python 3.13 sys_agent.py`; never approve a command: answer any approval prompt with `q`, point SYS_AUDIT_LOG at a temp copy, confirm no audit record has a `returncode`, clean up /tmp). List only the checks that truly need a human.
8. Do not tag. Wait for my explicit instruction.
