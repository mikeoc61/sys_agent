#!/usr/bin/env python3
"""
Regenerate the colorized example-session block in README.md.

The README's example session uses a GitHub ```ansi fenced block containing
literal ANSI escape bytes — which are invisible and painful to hand-edit.
This script is the maintainable source: edit EXAMPLE_LINES and COLOR_SCHEME
below in plain text, run the script, and it rewrites the block in-place.

Usage:
    python3 tools/gen_readme_example.py            # rewrite README.md
    python3 tools/gen_readme_example.py --check     # exit 1 if out of date
    python3 tools/gen_readme_example.py --print     # print block, don't write

The block is delimited in README.md by these HTML-comment markers:
    <!-- BEGIN EXAMPLE -->
    ```ansi
    ...
    ```
    <!-- END EXAMPLE -->
Markers let the script find and replace exactly the example, nothing else.
"""

from __future__ import annotations

import argparse
import os
import sys

# --- ANSI palette (mirrors sys_agent's actual color helpers) -----------------

E = "\033"
RESET   = f"{E}[0m"
B_CYAN  = f"{E}[1;36m"   # banner line, you> prefix
B_GREEN = f"{E}[1;32m"   # agent> prefix
B_BYEL  = f"{E}[1;93m"   # COMMAND: label
B_YEL   = f"{E}[1;33m"   # Run? approval prompt
GREEN   = f"{E}[32m"     # [exit=0]
RED     = f"{E}[31m"     # [exit=N]
YEL     = f"{E}[33m"     # [stderr] label
DIM     = f"{E}[2m"      # meta line, CWD, separators, [loaded ...]

SEP = "─" * 60

# --- Example content ---------------------------------------------------------
# Each entry is (color_or_None, text). color=None means plain (uncolored).
# Partial-line coloring (e.g. a colored prefix + plain body) is expressed as
# a tuple of segments; see the you>/COMMAND:/agent> lines.

EXAMPLE_LINES: list = [
    (None, "$ sys_agent"),
    (DIM,  "[loaded 2 vars from ~/.config/sys_agent/.env]"),
    (B_CYAN, "sys_agent  provider=anthropic  model=claude-haiku-4-5-20251001  "
             "host=pi5 (Linux/aarch64)"),
    (DIM,  "meta: /help  /info  /reset  /auto on|off  /tokens on|off  "
           "/color on|off  /exit, /quit   — /help for details"),
    (None, ""),
    [(B_CYAN, "you>"), (None, " what's eating disk on /var?")],
    (None, ""),
    (DIM,  SEP),
    [(B_BYEL, "COMMAND:"), (None, "  sudo du -sh /var/* 2>/dev/null | "
                                  "sort -rh | head -10")],
    (None, "REASON:   List the largest top-level subdirectories under /var"),
    (DIM,  "CWD:      /home/mikeoc"),
    (DIM,  SEP),
    [(B_YEL, "Run? [y]es / [n]o / [e]dit / [q]uit:"), (None, " y")],
    (GREEN, "[exit=0]"),
    (None, "1.4G  /var/lib"),
    (None, "820M  /var/log"),
    (None, "..."),
    (None, ""),
    [(B_GREEN, "agent>"), (None, " /var/lib dominates at 1.4G. "
                                 "Want me to break that down next?")],
]

BEGIN_MARKER = "<!-- BEGIN EXAMPLE -->"
END_MARKER = "<!-- END EXAMPLE -->"


def render_line(entry) -> str:
    """Render one EXAMPLE_LINES entry to a string with ANSI codes."""
    if isinstance(entry, list):
        # Multi-segment line: color each segment independently.
        out = []
        for color, text in entry:
            out.append(f"{color}{text}{RESET}" if color else text)
        return "".join(out)
    color, text = entry
    if color is None or text == "":
        return text
    return f"{color}{text}{RESET}"


def build_block() -> str:
    """Build the full ```ansi fenced block, marker-wrapped."""
    body = "\n".join(render_line(e) for e in EXAMPLE_LINES)
    return f"{BEGIN_MARKER}\n```ansi\n{body}\n```\n{END_MARKER}"


def readme_path() -> str:
    # Script lives in tools/; README.md is one level up.
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "README.md"))


def splice(content: str, block: str) -> str:
    """Replace the marker-delimited region in content with block."""
    try:
        start = content.index(BEGIN_MARKER)
        end = content.index(END_MARKER) + len(END_MARKER)
    except ValueError:
        sys.exit(
            f"error: markers {BEGIN_MARKER} / {END_MARKER} not found in "
            f"README.md — add them around the example block first."
        )
    return content[:start] + block + content[end:]


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate README example block.")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if README is out of date; don't write")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="print the generated block; don't write")
    args = ap.parse_args()

    block = build_block()

    if args.print_only:
        print(block)
        return

    path = readme_path()
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    updated = splice(content, block)

    if args.check:
        if content != updated:
            print("README.md example block is OUT OF DATE — run without --check")
            sys.exit(1)
        print("README.md example block is up to date")
        return

    if content == updated:
        print("README.md already up to date; nothing written")
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"README.md example block regenerated ({block.count(E)} escape codes)")


if __name__ == "__main__":
    main()
