#!/usr/bin/env python3
"""
Regenerate the example-session graphic in the README.

Renders a terminal-styled SVG (assets/example-session.svg) showing a sample
sys_agent session with the tool's actual color scheme. SVG is the only
format that renders color reliably across GitHub, Safari, Chrome, and
non-GitHub Markdown viewers — ANSI code blocks do not render in rendered
READMEs.

Edit EXAMPLE_LINES (plain text + color names) below, then run this script.
It rewrites the SVG and ensures the README references it.

Usage:
    python3 tools/gen_readme_example.py            # write SVG + fix README ref
    python3 tools/gen_readme_example.py --check     # exit 1 if SVG out of date
    python3 tools/gen_readme_example.py --print     # print SVG to stdout only

The README image reference is delimited by:
    <!-- BEGIN EXAMPLE -->
    ![...](assets/example-session.svg)
    <!-- END EXAMPLE -->
"""

from __future__ import annotations

import argparse
import html
import os
import sys

# --- Color scheme (hex; mirrors sys_agent's ANSI palette intent) -------------

COLORS = {
    "plain":   "#d4d4d4",   # default foreground
    "cyan":    "#4ec9d4",   # banner, you> prefix          (bold)
    "green":   "#6ac779",   # agent> prefix                (bold)
    "byellow": "#e5c07b",   # COMMAND: label               (bold, bright)
    "yellow":  "#d7ba7d",   # Run? prompt                  (bold)
    "exit_ok": "#6ac779",   # [exit=0]
    "exit_no": "#e06c75",   # [exit=N]
    "dim":     "#7a7a7a",   # meta line, CWD, separators, [loaded ...]
}
BACKGROUND = "#1e1e1e"
WINDOW_BAR = "#2d2d2d"
TRAFFIC = ("#ff5f56", "#ffbd2e", "#27c93f")

# --- Example content ---------------------------------------------------------
# Each line is a list of (color_key, bold, text) segments. A single-segment
# line is still a one-element list.

EXAMPLE_LINES: list[list[tuple[str, bool, str]]] = [
    [("plain", False, "$ sys_agent")],
    [("dim", False, "[loaded 2 vars from ~/.config/sys_agent/.env]")],
    [("cyan", True, "sys_agent  provider=anthropic  "
                    "model=claude-haiku-4-5-20251001  host=pi5 (Linux/aarch64)")],
    [("dim", False, "meta: /help  /info  /reset  /auto on|off  /tokens on|off  "
                    "/color on|off  /exit, /quit")],
    [("plain", False, "")],
    [("cyan", True, "you>"), ("plain", False, " what's eating disk on /var?")],
    [("plain", False, "")],
    [("dim", False, "\u2500" * 58)],
    [("byellow", True, "COMMAND:"),
     ("plain", False, "  sudo du -sh /var/* 2>/dev/null | sort -rh | head -10")],
    [("plain", False, "REASON:   List the largest top-level subdirs under /var")],
    [("dim", False, "CWD:      /home/mikeoc")],
    [("dim", False, "\u2500" * 58)],
    [("yellow", True, "Run? [y]es / [n]o / [e]dit / [q]uit:"),
     ("plain", False, " y")],
    [("exit_ok", False, "[exit=0]")],
    [("plain", False, "1.4G  /var/lib")],
    [("plain", False, "820M  /var/log")],
    [("plain", False, "...")],
    [("plain", False, "")],
    [("green", True, "agent>"),
     ("plain", False, " /var/lib dominates at 1.4G. "
                      "Want me to break that down next?")],
]

BEGIN_MARKER = "<!-- BEGIN EXAMPLE -->"
END_MARKER = "<!-- END EXAMPLE -->"
SVG_REL_PATH = "assets/example-session.svg"

# --- Layout constants --------------------------------------------------------

FONT = ("ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, "
        "'Liberation Mono', monospace")
FONT_SIZE = 14
LINE_H = 21
PAD_X = 18
PAD_TOP = 44          # room for the window title bar
PAD_BOTTOM = 16
BAR_H = 28
RADIUS = 8


def build_svg() -> str:
    # Canvas width: estimate from the longest line's char count. Used ONLY
    # for the SVG viewport size — segment positioning is handled by inline
    # tspan flow, so a loose estimate here just adds harmless right margin.
    APPROX_CHAR_W = 8.6
    cols = max(sum(len(seg[2]) for seg in line) for line in EXAMPLE_LINES)
    width = int(PAD_X * 2 + cols * APPROX_CHAR_W)
    height = int(PAD_TOP + len(EXAMPLE_LINES) * LINE_H + PAD_BOTTOM)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="{FONT}" font-size="{FONT_SIZE}">'
    )
    # Window background
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" '
        f'rx="{RADIUS}" fill="{BACKGROUND}"/>'
    )
    # Title bar
    parts.append(
        f'<path d="M0 {RADIUS} Q0 0 {RADIUS} 0 H{width - RADIUS} '
        f'Q{width} 0 {width} {RADIUS} V{BAR_H} H0 Z" fill="{WINDOW_BAR}"/>'
    )
    # Traffic-light dots
    for i, dot in enumerate(TRAFFIC):
        cx = PAD_X + i * 20
        parts.append(f'<circle cx="{cx}" cy="{BAR_H // 2}" r="6" fill="{dot}"/>')
    # Title text
    parts.append(
        f'<text x="{width // 2}" y="{BAR_H // 2 + 4}" fill="{COLORS["dim"]}" '
        f'text-anchor="middle" font-size="12">sys_agent</text>'
    )
    # Body lines — each line is ONE <text> element; segments are <tspan>
    # children that flow inline automatically. No manual x-positioning, no
    # width estimation, so colored segments can never overlap.
    for row, line in enumerate(EXAMPLE_LINES):
        y = PAD_TOP + row * LINE_H
        spans: list[str] = []
        for color_key, bold, text in line:
            if not text:
                continue
            weight = ' font-weight="bold"' if bold else ""
            esc = html.escape(text).replace(" ", "\u00a0")
            spans.append(
                f'<tspan fill="{COLORS[color_key]}"{weight}>{esc}</tspan>'
            )
        if not spans:
            continue
        parts.append(
            f'<text x="{PAD_X}" y="{y}" xml:space="preserve">'
            + "".join(spans) + "</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, ".."))


def ensure_readme_ref(root: str) -> None:
    """Make sure README's marker region points at the SVG."""
    path = os.path.join(root, "README.md")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    ref = (f"{BEGIN_MARKER}\n"
           f"![sys_agent example session]({SVG_REL_PATH})\n"
           f"{END_MARKER}")
    try:
        start = content.index(BEGIN_MARKER)
        end = content.index(END_MARKER) + len(END_MARKER)
    except ValueError:
        sys.exit(f"error: {BEGIN_MARKER}/{END_MARKER} markers not found in "
                 f"README.md — add them around the example block first.")
    updated = content[:start] + ref + content[end:]
    if updated != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
        print("README.md image reference updated")


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate README example SVG.")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the SVG on disk is stale; don't write")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="print SVG to stdout; don't write anything")
    args = ap.parse_args()

    svg = build_svg()

    if args.print_only:
        sys.stdout.write(svg)
        return

    root = repo_root()
    svg_path = os.path.join(root, SVG_REL_PATH)
    os.makedirs(os.path.dirname(svg_path), exist_ok=True)

    existing = ""
    if os.path.isfile(svg_path):
        with open(svg_path, "r", encoding="utf-8") as f:
            existing = f.read()

    if args.check:
        if existing != svg:
            print(f"{SVG_REL_PATH} is OUT OF DATE — run without --check")
            sys.exit(1)
        print(f"{SVG_REL_PATH} is up to date")
        return

    if existing == svg:
        print(f"{SVG_REL_PATH} already up to date")
    else:
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"{SVG_REL_PATH} regenerated")
    ensure_readme_ref(root)


if __name__ == "__main__":
    main()
