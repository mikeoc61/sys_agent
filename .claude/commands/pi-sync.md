---
description: Push main, sync pibot's checkout, run offline checks there
---
1. Run `git status`. Stop if the tree is dirty or main is not ahead of origin/main.
2. Run the local baseline: `python3 -m py_compile sys_agent.py && python3 test_consult_render.py`. Stop on failure.
3. Run `git push origin main`.
4. Run `ssh -o BatchMode=yes pibot 'cd ~/projects/sys_agent && git pull --ff-only && git log -1 --oneline && python3 -m py_compile sys_agent.py && python3 test_consult_render.py'`.
5. Confirm pibot's HEAD matches local HEAD (`git rev-parse HEAD`). Report both hashes and the harness RESULT line.
6. If everything passed, list the interactive checks I should run on pibot for this change, derived from `git diff $(git describe --tags --abbrev=0)..HEAD`.
7. Do not tag. Wait for my explicit instruction.
