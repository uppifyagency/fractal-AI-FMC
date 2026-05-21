"""md2tex.py — minimal markdown -> LaTeX converter for the paper draft.

Handles: headers, bold, italics, inline math, display math (preserves
$...$ and $$...$$), code spans, lists (- and 1.), tables (pipe-style),
links, blockquotes. Not a full pandoc replacement, but enough for a
self-contained academic prose markdown.

Usage:
    python md2tex.py draft.md > body.tex
"""
from __future__ import annotations
import re
import sys
from pathlib import Path


def convert_table(rows: list[str]) -> str:
    """Pipe-style table -> LaTeX tabular."""
    if len(rows) < 2:
        return "\n".join(rows)
    # Strip leading/trailing pipes, split cells
    parsed = []
    for r in rows:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        parsed.append(cells)
    n_cols = len(parsed[0])
    # Skip the alignment row (---)
    body = [parsed[0]] + parsed[2:]
    # Build LaTeX
    align = " ".join(["l"] * n_cols)
    out = ["\\begin{tabular}{" + align + "}", "\\toprule"]
    for i, row in enumerate(body):
        # pad/truncate to n_cols
        row = (row + [""] * n_cols)[:n_cols]
        out.append(" & ".join(escape_cell(c) for c in row) + " \\\\")
        if i == 0:
            out.append("\\midrule")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def escape_cell(s: str) -> str:
    """Escape special LaTeX chars in table cells, preserve inline math."""
    return inline_md_to_tex(s)


def inline_md_to_tex(s: str) -> str:
    """Convert inline markdown to LaTeX. Preserves $...$ math."""
    # Save math spans
    math_spans: list[str] = []

    def save_math(m):
        idx = len(math_spans)
        math_spans.append(m.group(0))
        return f"@@MATH{idx}@@"

    # Save display math first
    s = re.sub(r"\$\$(.+?)\$\$", save_math, s, flags=re.S)
    # Then inline math
    s = re.sub(r"\$([^\$]+?)\$", save_math, s)

    # Code spans
    s = re.sub(r"`([^`]+)`", lambda m: r"\texttt{" + escape_text(m.group(1)) + "}", s)
    # Bold **x** or __x__
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"__(.+?)__", r"\\textbf{\1}", s)
    # Italic *x*
    s = re.sub(r"(?<!\*)\*([^\*]+?)\*(?!\*)", r"\\emph{\1}", s)
    # Links [text](url)
    s = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)",
               lambda m: r"\href{" + m.group(2) + "}{" + m.group(1) + "}", s)
    # Escape remaining special chars (carefully — leave \, {, } alone for math/cmds)
    # Just escape & % # _
    s = re.sub(r"(?<!\\)&", r"\\&", s)
    s = re.sub(r"(?<!\\)%", r"\\%", s)
    s = re.sub(r"(?<!\\)#", r"\\#", s)
    s = re.sub(r"(?<!\\)_(?![a-zA-Z]*\})", r"\\_", s)
    # Restore math
    for idx, math in enumerate(math_spans):
        s = s.replace(f"@@MATH{idx}@@", math)
    return s


def escape_text(s: str) -> str:
    """Plain text escape — for code spans, no markdown interpretation."""
    return (s.replace("\\", r"\textbackslash ")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("$", r"\$"))


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    in_table = False
    table_buf: list[str] = []
    in_code_block = False

    def flush_table():
        nonlocal in_table, table_buf
        if table_buf:
            out.append(convert_table(table_buf))
            table_buf = []
        in_table = False

    while i < len(lines):
        line = lines[i]

        # Code block
        if line.strip().startswith("```"):
            flush_table()
            in_code_block = not in_code_block
            out.append(r"\begin{verbatim}" if in_code_block else r"\end{verbatim}")
            i += 1
            continue
        if in_code_block:
            out.append(line)
            i += 1
            continue

        # Display math (single-line $$ ... $$)
        m = re.match(r"^\s*\$\$\s*$", line)
        if m:
            flush_table()
            out.append(r"\[")
            i += 1
            while i < len(lines) and not re.match(r"^\s*\$\$\s*$", lines[i]):
                out.append(lines[i])
                i += 1
            out.append(r"\]")
            i += 1
            continue
        # Inline-display math: $$...$$ on one line
        if "$$" in line and line.count("$$") == 2:
            flush_table()
            out.append(line.replace("$$", r"\[", 1).replace("$$", r"\]", 1))
            i += 1
            continue

        # Tables (pipe-prefixed, with --- separator on row 2)
        if line.strip().startswith("|") and "|" in line:
            if not in_table:
                # Lookahead for alignment row
                if (i + 1 < len(lines)
                        and re.match(r"^\s*\|[\s\-:|]+\|\s*$", lines[i+1])):
                    in_table = True
                    table_buf = [line]
                    i += 1
                    continue
            else:
                table_buf.append(line)
                i += 1
                continue
        elif in_table:
            flush_table()

        # Headers
        m = re.match(r"^(#{1,4})\s+(.+)$", line)
        if m:
            flush_table()
            level, title = len(m.group(1)), m.group(2).strip()
            cmd = {1: "section", 2: "section", 3: "subsection", 4: "subsubsection"}[min(level, 4)]
            out.append(f"\\{cmd}{{{inline_md_to_tex(title)}}}")
            i += 1
            continue

        # Block quote
        m = re.match(r"^>\s?(.*)$", line)
        if m:
            flush_table()
            out.append(r"\begin{quote}")
            out.append(inline_md_to_tex(m.group(1)))
            j = i + 1
            while j < len(lines) and lines[j].startswith(">"):
                out.append(inline_md_to_tex(re.sub(r"^>\s?", "", lines[j])))
                j += 1
            out.append(r"\end{quote}")
            i = j
            continue

        # List items
        m_ul = re.match(r"^(\s*)[-*]\s+(.+)$", line)
        m_ol = re.match(r"^(\s*)\d+\.\s+(.+)$", line)
        if m_ul or m_ol:
            flush_table()
            env = "itemize" if m_ul else "enumerate"
            out.append(f"\\begin{{{env}}}")
            j = i
            indent_base = (m_ul or m_ol).group(1)
            current_item: list[str] = []
            def flush_item():
                if current_item:
                    text = " ".join(s.strip() for s in current_item)
                    out.append(f"\\item {inline_md_to_tex(text)}")
                    current_item.clear()
            while j < len(lines):
                cur = lines[j]
                m_ul_j = re.match(r"^(\s*)[-*]\s+(.+)$", cur)
                m_ol_j = re.match(r"^(\s*)\d+\.\s+(.+)$", cur)
                # New item at same indent: flush prev, start new
                if (m_ul_j or m_ol_j) and (m_ul_j or m_ol_j).group(1) == indent_base:
                    flush_item()
                    current_item.append((m_ul_j or m_ol_j).group(2))
                    j += 1
                # Blank line: end-of-list candidate; lookahead
                elif cur.strip() == "":
                    nxt = lines[j+1] if j+1 < len(lines) else ""
                    nxt_is_item = (re.match(r"^(\s*)[-*]\s+(.+)$", nxt) or
                                   re.match(r"^(\s*)\d+\.\s+(.+)$", nxt))
                    if nxt_is_item and nxt_is_item.group(1) == indent_base:
                        # blank between items — just continue
                        j += 1
                    else:
                        # end of list
                        j += 1
                        break
                # Continuation line (indented or just text under item): append
                elif current_item and (cur.startswith("   ") or cur[:1] not in "#"):
                    current_item.append(cur)
                    j += 1
                else:
                    break
            flush_item()
            out.append(f"\\end{{{env}}}")
            i = j
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}\s*$", line):
            flush_table()
            out.append(r"\hrulefill")
            i += 1
            continue

        # Paragraph / blank
        if line.strip() == "":
            flush_table()
            out.append("")
        else:
            out.append(inline_md_to_tex(line))
        i += 1

    flush_table()
    return "\n".join(out)


def main():
    if len(sys.argv) < 2:
        print("usage: md2tex.py input.md [> output.tex]", file=sys.stderr)
        sys.exit(1)
    md = Path(sys.argv[1]).read_text()
    print(convert(md))


if __name__ == "__main__":
    main()
