"""Render the README benchmark figure from the published measurements."""

import argparse
from io import BytesIO
import json
from pathlib import Path

import cairosvg
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
BG = "#101619"
GRID = "#2a383d"
INK = "#f7f5ed"
MUTED = "#b2c1c4"
YELLOW = "#ffe32b"
TEAL = "#75d4d9"


def text(figure, x, y, value, size, *, color=INK, weight="normal", **kwargs):
    return figure.text(x, y, value, fontsize=size, color=color, fontweight=weight, **kwargs)


def brand(figure):
    logo = ROOT / "demo_chat/static/lightreason.svg"
    raster = cairosvg.svg2png(url=str(logo), output_width=800)
    axes = figure.add_axes((0.039, 0.909, 0.197, 0.069), frameon=False)
    axes.imshow(mpimg.imread(BytesIO(raster), format="png"))
    axes.set_axis_off()
    text(figure, 0.956, 0.944, "ORIGINAL BF16 WEIGHTS   /   BATCH 1", 9.4, color=MUTED,
         ha="right", va="center")


def panel(figure, data, bounds, color):
    x, y, w, h = bounds
    text(figure, x, y + h - 0.024, data["title"], 17.2, weight="semibold")
    text(figure, x, y + h - 0.058, data["subtitle"], 8.7, color=MUTED)
    plot_left, plot_width = 0.460, 0.270
    rate_column, ratio_column = 0.865, 0.998
    column_y = y + h - 0.107
    text(figure, x + w * rate_column, column_y, "TOK/S", 7.6, color=MUTED, ha="right", weight="semibold")
    text(figure, x + w * ratio_column, column_y, "× INITIAL", 7.6, color=MUTED, ha="right", weight="semibold")
    ax = figure.add_axes((x + w * plot_left, y + 0.06, w * plot_width, h - 0.176), facecolor="none")
    ax.set_xlim(*data["limits"])
    ax.set_xticks(data["ticks"])
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.4, length=0, pad=8)
    rows = data["points"]
    ax.set_ylim(-0.62, len(rows) - 0.40)
    ax.set_yticks([])
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    transform = ax.get_yaxis_transform()
    for i, row in enumerate(rows):
        k = len(rows) - 1 - i
        ax.text(-plot_left / plot_width, k + 0.13, row["label"], transform=transform,
                va="center", fontsize=9.6, fontweight="semibold", color=INK, clip_on=False)
        ax.text(-plot_left / plot_width, k - 0.22, row["context"], transform=transform,
                va="center", fontsize=7.9, color=MUTED, clip_on=False)
        filled = row["marker"] == "filled"
        reference = row["marker"] == "reference"
        row_color = MUTED if reference else color
        ax.hlines(k, data["limits"][0], row["value"], lw=2.6, color=row_color, alpha=0.55, zorder=2)
        ax.plot(row["value"], k, marker="s" if reference else "o", markersize=6.1,
                markerfacecolor=MUTED if reference else color if filled else BG,
                markeredgecolor=row_color, markeredgewidth=1.25, zorder=3)
        rate = format(row["value"], f'.{data["rate_decimals"]}f')
        ratio = row["value"] / row["reference_value"]
        ax.text((rate_column - plot_left) / plot_width, k, rate, transform=transform,
                ha="right", va="center", color=INK, fontsize=10.4, fontweight="semibold", clip_on=False)
        ax.text((ratio_column - plot_left) / plot_width, k, f"{ratio:.2f}×", transform=transform,
                ha="right", va="center", color=row_color, fontsize=10.4, fontweight="semibold", clip_on=False)
    text(figure, x + w * plot_left, y + 0.008, "Tokens per second", 8.4, color=MUTED)


def render(data, output, font):
    plt.switch_backend("Agg")
    family = "DejaVu Sans"
    if font is not None:
        fontManager.addfont(font)
        family = FontProperties(fname=font).get_name()
    mpl.rcParams.update({"font.family": family, "svg.fonttype": "path", "svg.hashsalt": "lightreason-benchmark",
                         "savefig.facecolor": BG, "figure.facecolor": BG})
    figure = plt.figure(figsize=(12.8, 7.2))
    brand(figure)
    text(figure, 0.042, 0.838, "How fast can Qwen run on one RTX 4090?", 28.5, weight="bold")
    text(figure, 0.044, 0.793, "Single-request decoding after the first token", 11,
         color=MUTED)
    figure.add_artist(Line2D((0.5, 0.5), (0.184, 0.732), color=GRID, linewidth=0.85))
    panel(figure, data["panels"][0], (0.044, 0.172, 0.425, 0.574), YELLOW)
    panel(figure, data["panels"][1], (0.531, 0.172, 0.425, 0.574), TEAL)
    text(figure, 0.044, 0.107, "■  Initial runner     ●  Exact IDs and stop vs. initial     ○  IDs differ or parity unverified",
         9.1, color=INK)
    text(figure, 0.044, 0.065, "Each ratio uses its matching prompt or suite. Panels use different scales. 35B milestones span separate sessions.",
         8.5, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, metadata={"Title": "Lightreason benchmark history",
                                             "Description": "Qwen2.5-3B and Qwen3.6-35B-A3B on one RTX 4090"})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "docs/data/speedup-history.json")
    parser.add_argument("--out", type=Path, default=ROOT / "assets/speedup-history.png")
    parser.add_argument("--font", type=Path, help="Optional sans-serif font file")
    args = parser.parse_args()
    render(json.loads(args.data.read_text()), args.out, args.font)


if __name__ == "__main__":
    main()
