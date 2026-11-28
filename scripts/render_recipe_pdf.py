"""Render recipe-book.md to a clean, readable recipe-book.pdf using matplotlib."""
from pathlib import Path
import re
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "recipe-book.md"
PDF_PATH = ROOT / "recipe-book.pdf"

PAGE_W, PAGE_H = 8.27, 11.69  # A4 inches
MARGIN_X = 0.70
MARGIN_TOP = 0.70
MARGIN_BOT = 0.65
USABLE_W = PAGE_W - 2 * MARGIN_X


def strip_md_inline(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return text


def parse_blocks(md_text: str):
    blocks = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped == "---":
            blocks.append(("hr", ""))
            i += 1
        elif stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            blocks.append(("code", "\n".join(code_lines)))
        elif stripped.startswith("# "):
            blocks.append(("h1", strip_md_inline(stripped[2:].strip())))
            i += 1
        elif stripped.startswith("## "):
            blocks.append(("h2", strip_md_inline(stripped[3:].strip())))
            i += 1
        elif stripped.startswith("### "):
            blocks.append(("h3", strip_md_inline(stripped[4:].strip())))
            i += 1
        elif stripped.startswith("- "):
            blocks.append(("bullet", strip_md_inline(stripped[2:].strip())))
            i += 1
        else:
            para = [stripped]
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if not nxt or nxt.startswith(("#", "```", "- ", "---")):
                    break
                para.append(nxt)
                i += 1
            raw = " ".join(para)
            is_label = raw.startswith("**") and raw.endswith("**")
            blocks.append(("label" if is_label else "p", strip_md_inline(raw)))
    return blocks


def render_pdf(md_path: Path = MD_PATH, pdf_path: Path = PDF_PATH):
    blocks = parse_blocks(md_path.read_text(encoding="utf-8"))
    pages = []

    def new_page():
        fig = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, PAGE_W)
        ax.set_ylim(0, PAGE_H)
        ax.axis("off")
        pages.append((fig, ax))
        return ax, PAGE_H - MARGIN_TOP

    ax, y = new_page()

    def ensure_space(needed: float):
        nonlocal ax, y
        if y - needed < MARGIN_BOT:
            ax, y = new_page()

    for kind, content in blocks:
        if kind == "hr":
            ensure_space(0.20)
            y -= 0.06
            ax.plot([MARGIN_X, PAGE_W - MARGIN_X], [y, y], color="#d1d5db", lw=0.8)
            y -= 0.14
        elif kind == "h1":
            ensure_space(0.55)
            ax.text(MARGIN_X, y, content, fontsize=17, fontweight="bold", color="#111827", va="top", family="DejaVu Sans")
            y -= 0.34
            ax.plot([MARGIN_X, PAGE_W - MARGIN_X], [y, y], color="#2563eb", lw=1.6)
            y -= 0.18
        elif kind == "h2":
            ensure_space(0.50)
            y -= 0.12
            ax.text(MARGIN_X, y, content, fontsize=13, fontweight="bold", color="#1e3a8a", va="top", family="DejaVu Sans")
            y -= 0.25
            ax.plot([MARGIN_X, PAGE_W - MARGIN_X], [y, y], color="#e5e7eb", lw=0.9)
            y -= 0.12
        elif kind == "h3":
            ensure_space(0.35)
            y -= 0.08
            ax.text(MARGIN_X, y, content, fontsize=10.5, fontweight="bold", color="#1f2937", va="top", family="DejaVu Sans")
            y -= 0.22
        elif kind == "label":
            ensure_space(0.24)
            ax.text(MARGIN_X, y, content, fontsize=9.2, fontweight="bold", color="#374151", va="top", family="DejaVu Sans")
            y -= 0.19
        elif kind == "p":
            wrapped = textwrap.wrap(content, width=95)
            line_h = 0.185
            needed = len(wrapped) * line_h + 0.10
            ensure_space(needed)
            for wl in wrapped:
                ax.text(MARGIN_X, y, wl, fontsize=9.2, color="#1f2937", va="top", family="DejaVu Sans")
                y -= line_h
            y -= 0.08
        elif kind == "bullet":
            wrapped = textwrap.wrap(content, width=90)
            line_h = 0.185
            needed = len(wrapped) * line_h + 0.06
            ensure_space(needed)
            for idx, wl in enumerate(wrapped):
                prefix = "• " if idx == 0 else "  "
                ax.text(MARGIN_X + 0.12, y, prefix + wl, fontsize=9.2, color="#1f2937", va="top", family="DejaVu Sans")
                y -= line_h
            y -= 0.05
        elif kind == "code":
            raw_lines = content.splitlines()
            code_lines = []
            for rl in raw_lines:
                if len(rl) <= 84:
                    code_lines.append(rl)
                else:
                    code_lines.extend(textwrap.wrap(rl, width=84, subsequent_indent="    "))
            line_h = 0.165
            pad_y = 0.10
            box_h = len(code_lines) * line_h + 2 * pad_y
            ensure_space(box_h + 0.14)
            rect = FancyBboxPatch(
                (MARGIN_X, y - box_h),
                USABLE_W,
                box_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="#f3f4f6",
                edgecolor="#d1d5db",
                linewidth=0.8,
            )
            ax.add_patch(rect)
            cy = y - pad_y
            for cl in code_lines:
                ax.text(MARGIN_X + 0.12, cy, cl, fontsize=8.0, color="#111827", va="top", family="DejaVu Sans Mono")
                cy -= line_h
            y -= box_h + 0.14

    total = len(pages)
    with PdfPages(pdf_path) as pdf:
        for idx, (fig, pax) in enumerate(pages, start=1):
            pax.text(PAGE_W / 2, 0.35, f"Page {idx} of {total}", fontsize=8, color="#6b7280", ha="center", family="DejaVu Sans")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Rendered {pdf_path} ({total} pages, {pdf_path.stat().st_size} bytes)")


if __name__ == "__main__":
    render_pdf()
