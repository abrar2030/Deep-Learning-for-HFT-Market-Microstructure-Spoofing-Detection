"""
Console output formatting for pipeline runs.

Zero-dependency, pure-stdlib helpers producing clean, aligned terminal
output: boxed banners, numbered section rules, aligned tables, key-value
panels, and PASS / FAIL badges. ANSI colors are applied only when the
stream is a TTY and NO_COLOR is unset, so piped logs and CI output stay
plain. No emojis are used anywhere.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Dict, Iterable, List, Optional, Sequence

# --------------------------------------------------------------------- #
# Color handling
# --------------------------------------------------------------------- #


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()


class _C:
    RESET = "\033[0m" if _COLOR else ""
    BOLD = "\033[1m" if _COLOR else ""
    DIM = "\033[2m" if _COLOR else ""
    BLUE = "\033[38;5;33m" if _COLOR else ""
    CYAN = "\033[38;5;44m" if _COLOR else ""
    GREEN = "\033[38;5;35m" if _COLOR else ""
    YELLOW = "\033[38;5;178m" if _COLOR else ""
    RED = "\033[38;5;160m" if _COLOR else ""
    GRAY = "\033[38;5;245m" if _COLOR else ""


def _width(minimum: int = 64, maximum: int = 96) -> int:
    try:
        cols = shutil.get_terminal_size().columns
    except Exception:
        cols = 80
    return max(minimum, min(cols - 2, maximum))


def _visible_len(s: str) -> int:
    """Length of a string ignoring ANSI escape sequences."""
    out, i, n = 0, 0, len(s)
    while i < n:
        if s[i] == "\033":
            while i < n and s[i] != "m":
                i += 1
            i += 1
        else:
            out += 1
            i += 1
    return out


# --------------------------------------------------------------------- #
# Banners and sections
# --------------------------------------------------------------------- #


def banner(title: str, subtitle: str = "", meta: Optional[Dict] = None) -> None:
    """Boxed run header with optional subtitle and metadata lines."""
    w = _width()
    inner = w - 2
    lines: List[str] = []
    lines.append(f"{_C.BLUE}{_C.BOLD}" + "\u250c" + "\u2500" * inner + "\u2510")
    lines.append(
        "\u2502"
        + f"{_C.RESET}{_C.BOLD}"
        + title.center(inner)
        + f"{_C.RESET}{_C.BLUE}{_C.BOLD}\u2502"
    )
    if subtitle:
        lines.append(
            "\u2502"
            + f"{_C.RESET}{_C.GRAY}"
            + subtitle.center(inner)
            + f"{_C.RESET}{_C.BLUE}{_C.BOLD}\u2502"
        )
    if meta:
        lines.append("\u251c" + "\u2500" * inner + "\u2524")
        for k, v in meta.items():
            body = f"  {k}: {v}"
            pad = inner - _visible_len(body)
            lines.append(
                "\u2502"
                + f"{_C.RESET}{_C.GRAY}"
                + body
                + " " * max(pad, 0)
                + f"{_C.RESET}{_C.BLUE}{_C.BOLD}\u2502"
            )
    lines.append("\u2514" + "\u2500" * inner + "\u2518" + _C.RESET)
    print("\n".join(lines))


def section(number, title: str) -> None:
    """Numbered section rule: `── 3 | TRAINING ────────`."""
    w = _width()
    label = f" {number} \u2502 {title.upper()} "
    fill = max(w - _visible_len(label) - 4, 4)
    print(
        f"\n{_C.CYAN}{_C.BOLD}\u2500\u2500\u2500\u2500{label}"
        + "\u2500" * fill
        + _C.RESET
    )


def rule(char: str = "\u2500") -> None:
    print(_C.GRAY + char * _width() + _C.RESET)


# --------------------------------------------------------------------- #
# Tables and panels
# --------------------------------------------------------------------- #


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence],
    highlight_row: Optional[int] = None,
    align: Optional[str] = None,
) -> None:
    """
    Aligned box-drawing table.

    align: string of 'l' / 'r' per column (default: first column left,
    the rest right, the usual layout for name + numbers).
    """
    rows = [[str(c) for c in r] for r in rows]
    headers = [str(h) for h in headers]
    ncol = len(headers)
    if align is None:
        align = "l" + "r" * (ncol - 1)
    widths = [
        (
            max(_visible_len(headers[i]), *(_visible_len(r[i]) for r in rows))
            if rows
            else _visible_len(headers[i])
        )
        for i in range(ncol)
    ]

    def _pad(cell, width, side):
        gap = " " * max(width - _visible_len(cell), 0)
        return cell + gap if side == "l" else gap + cell

    def fmt_row(cells, color="", bold=False):
        parts = [_pad(c, widths[i], align[i]) for i, c in enumerate(cells)]
        style = (_C.BOLD if bold else "") + color
        return "\u2502 " + style + " \u2502 ".join(parts) + _C.RESET + " \u2502"

    top = "\u250c" + "\u252c".join("\u2500" * (w + 2) for w in widths) + "\u2510"
    mid = "\u251c" + "\u253c".join("\u2500" * (w + 2) for w in widths) + "\u2524"
    bot = "\u2514" + "\u2534".join("\u2500" * (w + 2) for w in widths) + "\u2518"

    print(_C.GRAY + top + _C.RESET)
    print(fmt_row(headers, bold=True))
    print(_C.GRAY + mid + _C.RESET)
    for idx, r in enumerate(rows):
        color = _C.GREEN if idx == highlight_row else ""
        print(fmt_row(r, color=color, bold=idx == highlight_row))
    print(_C.GRAY + bot + _C.RESET)


def kv(label: str, value, indent: int = 2, label_width: int = 34) -> None:
    """Aligned key-value line with dot leaders."""
    label_s = str(label)
    dots = max(label_width - len(label_s), 2)
    print(
        " " * indent
        + f"{label_s} {_C.GRAY}"
        + "." * dots
        + f"{_C.RESET} {_C.BOLD}{value}{_C.RESET}"
    )


def badge(ok: bool, true_text: str = "PASS", false_text: str = "FAIL") -> str:
    """Colored inline status badge string."""
    if ok:
        return f"{_C.GREEN}{_C.BOLD}{true_text}{_C.RESET}"
    return f"{_C.RED}{_C.BOLD}{false_text}{_C.RESET}"


def note(text: str) -> None:
    print(f"  {_C.GRAY}{text}{_C.RESET}")


def summary_panel(title: str, items: Dict[str, object], footer: str = "") -> None:
    """Final results panel: boxed, key-value aligned."""
    w = _width()
    inner = w - 2
    print()
    print(f"{_C.GREEN}{_C.BOLD}\u250c" + "\u2500" * inner + "\u2510")
    print(
        "\u2502"
        + f"{_C.RESET}{_C.BOLD}"
        + title.center(inner)
        + f"{_C.RESET}{_C.GREEN}{_C.BOLD}\u2502"
    )
    print("\u251c" + "\u2500" * inner + "\u2524" + _C.RESET)
    label_w = min(max((len(str(k)) for k in items), default=10) + 2, 40)
    for k, v in items.items():
        body = f"  {str(k).ljust(label_w)} {v}"
        pad = inner - _visible_len(body)
        print(
            f"{_C.GREEN}{_C.BOLD}\u2502{_C.RESET}"
            + body
            + " " * max(pad, 0)
            + f"{_C.GREEN}{_C.BOLD}\u2502{_C.RESET}"
        )
    if footer:
        print(f"{_C.GREEN}{_C.BOLD}\u251c" + "\u2500" * inner + "\u2524" + _C.RESET)
        body = f"  {footer}"
        pad = inner - _visible_len(body)
        print(
            f"{_C.GREEN}{_C.BOLD}\u2502{_C.RESET}{_C.GRAY}"
            + body
            + " " * max(pad, 0)
            + f"{_C.RESET}{_C.GREEN}{_C.BOLD}\u2502{_C.RESET}"
        )
    print(f"{_C.GREEN}{_C.BOLD}\u2514" + "\u2500" * inner + "\u2518" + _C.RESET)


def step_done(text: str) -> None:
    """Completed-step line with a check-style marker (ASCII, no emoji)."""
    print(f"  {_C.GREEN}[ok]{_C.RESET} {text}")
