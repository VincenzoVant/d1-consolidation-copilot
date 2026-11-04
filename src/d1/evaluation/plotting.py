"""One chart style for every notebook in notebooks/.

Each model or method keeps one colour in every chart, so a reader who learned
"bge-m3 is orange" is never misled. The colours were checked for colour-blind
readability with a validator; light ones (aqua, yellow) always come with
visible labels or a table.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SURFACE, INK, INK_2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b4b3ad"

SHORT_NAMES = {
    "intfloat/multilingual-e5-base": "e5-base",
    "BAAI/bge-m3": "bge-m3",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet",
}
COLORS = {
    # embedding models
    "intfloat/multilingual-e5-base": "#2a78d6",
    "BAAI/bge-m3": "#eb6834",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "#1baf7a",
}
# The other entities that appear in charts, each with its own fixed colours.
SINGLE = "#2a78d6"  # a chart with one series
SOURCE_COLORS = {"rules": "#2a78d6", "llm": "#eb6834"}
LANGUAGE_COLORS = {"nl": "#2a78d6", "fr": "#eb6834", "en": "#1baf7a", "mixed": "#eda100"}
# The six mapping methods, in the reference order (validated for adjacent bars).
METHOD_COLORS = {
    "fuzzy": "#2a78d6", "embeddings": "#eb6834", "xgboost": "#1baf7a",
    "random_forest": "#eda100", "laya": "#e87ba4", "llm": "#008300",
}
# Sequential blue, light to dark, for heatmaps.
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def short(name: str) -> str:
    return SHORT_NAMES.get(name, name)


def pct(decimals: int = 0) -> PercentFormatter:
    return PercentFormatter(1.0, decimals=decimals)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 110,
    })  # fmt: skip
