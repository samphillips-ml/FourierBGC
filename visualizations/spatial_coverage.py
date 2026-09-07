"""
Spatial coverage figure for FourierBGC Data section (final).

Two-panel Mediterranean map: left panel = train profiles, right panel =
test profiles. Points are colored by target variable (BBP700/CHLA/NITRATE)
with a small random jitter applied so overlapping profiles (most floats
carry all three variables at the same lat/lon) partially separate visually
instead of fully occluding each other. PPCon's five evaluation sub-regions
(NWM/SWM/TYR/ION/LEV) are drawn as labeled dashed boxes on both panels,
since this paper's Results section reuses PPCon's own regional breakdown.

Style choices (deliberate, not defaults):
  - Colour legend sits inside the train panel, in the upper-right corner
    over the Black Sea, which falls inside the map extent but contains no
    profiles and no evaluation sub-region boxes. Copernicus figure
    guidance requires that a legend clarify all symbols *in the figure
    itself* rather than only in the caption text, so the colour key cannot
    live in the caption alone. Placing it in dead map area rather than
    below the panels avoids adding height to an already wide, short
    figure.
  - Colors: Okabe-Ito CVD-safe categorical palette (vermillion / blue /
    bluish green). These separate by lightness as well as hue, so they
    remain distinguishable under deuteranopia and protanopia, where a
    green/orange pairing would collapse into indistinguishable browns.
    Deliberately not all shades of green, since all three variables have
    some "naturally green" association (chlorophyll, nitrate-fed blooms,
    particulate-matter proxy) that would defeat color-coding if followed
    literally.
  - No political borders on the basemap — coastlines only, per Copernicus
    guidance on depoliticized maps.
  - Coordinate tick labels use a space before the direction (e.g. "30° N",
    "5° W") per Copernicus figure content guidelines.
  - Latitude tick labels only on the leftmost (train) panel, since both
    panels share the same extent and repeating them is just clutter.
  - Region labels sit in a semi-transparent white box, since region
    boundaries often fall over open water or dense point clusters with no
    uniform clean background to place text against.

Run from the project root:
    python visualizations/spatial_coverage.py

Expects data laid out as:
    data/BBP700/float_ds_sf_train.csv
    data/BBP700/float_ds_sf_test.csv
    data/CHLA/float_ds_sf_train.csv
    data/CHLA/float_ds_sf_test.csv
    data/NITRATE/float_ds_sf_train.csv
    data/NITRATE/float_ds_sf_test.csv

Each CSV is transposed: first column is a row label (year, day_rad, lat,
lon, temp, psal, doxy, <TARGET>), remaining columns are one profile each.

Output:
    visualizations/spatial_coverage.pdf  (vector — submission/Overleaf copy)
    visualizations/spatial_coverage.png  (600 dpi raster — preview/backup)

Overleaf usage: copy spatial_coverage.pdf into your Overleaf project's
figures folder (e.g. figures/spatial_coverage.pdf) and reference it with:

    \\begin{figure}[t]
        \\centering
        \\includegraphics[width=\\textwidth]{figures/spatial_coverage.pdf}
        \\caption{...}  % see suggested caption printed by this script
        \\label{fig:spatial_coverage}
    \\end{figure}

No conversion needed — it's already a standard vector PDF, which is what
Overleaf (pdfLaTeX) expects for \\includegraphics. Text inside the figure
(axis labels, region names, legend) is real vector text, not rasterized,
so it stays crisp and selectable at any zoom in the compiled PDF.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

# ---- config ----------------------------------------------------------

DATA_ROOT = Path("data")
OUT_DIR = Path("visualizations")

VARIABLES = ["BBP700", "CHLA", "NITRATE"]
SPLITS = ["train", "test"]  # 'removed' is excluded data, not plotted

# Mediterranean bounding box (lon_min, lon_max, lat_min, lat_max)
EXTENT = [-6, 37, 30, 46]

# Okabe-Ito colourblind-safe categorical palette.
# Chosen triple is maximally separated in both hue and lightness, so the
# categories stay distinguishable under deuteranopia/protanopia/tritanopia.
# Verify with the Coblis simulator before submission, as Copernicus asks.
VARIABLE_COLORS = {
    "CHLA": "#009E73",     # bluish green (Okabe-Ito)
    "NITRATE": "#D55E00",  # vermillion (Okabe-Ito)
    "BBP700": "#0072B2",   # blue (Okabe-Ito)
}
VARIABLE_LABELS = {
    "CHLA": "Chlorophyll-$a$",
    "NITRATE": "Nitrate",
    "BBP700": "bbp700",
}

# Order the legend entries to match how the variables are discussed in the
# text (nitrate, chlorophyll-a, bbp700) rather than the load order.
LEGEND_ORDER = ["NITRATE", "CHLA", "BBP700"]

# PPCon's own evaluation sub-regions, from utils_analysis.py dict_ga:
# {region: [[lat_min, lat_max], [lon_min, lon_max]]}
REGIONS = {
    "NWM": [[40, 45], [-2, 9.5]],
    "SWM": [[32, 40], [-2, 9.5]],
    "TYR": [[37, 45], [9.5, 16]],
    "ION": [[30, 37], [9.5, 22]],
    "LEV": [[30, 37], [22, 36]],
}

LAND_COLOR = "#e8e4d8"
OCEAN_COLOR = "#d4e6f1"
COAST_COLOR = "#555555"
REGION_BOX_COLOR = "#222222"

POINT_SIZE = 3
POINT_ALPHA = 0.65
JITTER_DEGREES = 0.08  # ~ a few km at Mediterranean latitudes

# Legend markers are drawn larger and fully opaque than the plotted points,
# since a 3 pt, 65%-alpha dot is not legible as a colour key.
LEGEND_MARKER_SIZE = 6
LEGEND_FONTSIZE = 8

# The map extent is roughly 43 deg lon by 16 deg lat, i.e. about 2.7:1, and
# cartopy holds that aspect fixed. Sizing the canvas to match avoids the
# large vertical dead space that appears if the figure is made taller than
# the maps can fill. Extra height beyond the maps is for titles and ticks.
FIG_WIDTH = 11
FIG_HEIGHT = 3.1

PANEL_TITLES = {"train": "Train", "test": "Test"}

# ---- data loading ------------------------------------------------------


def load_lat_lon(csv_path: Path) -> pd.DataFrame:
    """Load a transposed PPCon-format CSV and return just lat/lon per profile."""
    df = pd.read_csv(csv_path, header=None, index_col=0)
    scalars = df.loc[["lat", "lon"]].T
    scalars = scalars.apply(pd.to_numeric, errors="coerce")
    return scalars.dropna(subset=["lat", "lon"])


def load_all_splits() -> dict[str, dict[str, pd.DataFrame]]:
    """Load lat/lon for every (variable, split) combination.

    Returns {split: {variable: DataFrame}}.
    """
    out = {split: {} for split in SPLITS}
    for variable in VARIABLES:
        for split in SPLITS:
            csv_path = DATA_ROOT / variable / f"float_ds_sf_{split}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Expected file not found: {csv_path}")
            out[split][variable] = load_lat_lon(csv_path)
    return out


# ---- plotting ------------------------------------------------------


def _format_lon(x, _):
    """Longitude tick label, Copernicus style: degree sign + space + direction."""
    if x >= 0:
        return f"{x:.0f}\u00b0 E"
    return f"{-x:.0f}\u00b0 W"


def _format_lat(y, _):
    """Latitude tick label, Copernicus style: degree sign + space + direction."""
    if y >= 0:
        return f"{y:.0f}\u00b0 N"
    return f"{-y:.0f}\u00b0 S"


def add_basemap(ax, show_lat_labels):
    """Add coastline-only basemap. No political borders (Copernicus depoliticization guidance)."""
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    ax.coastlines(resolution="10m", linewidth=0.5, color=COAST_COLOR, zorder=1)

    # NOTE: draw_labels=True on gridlines triggers a known cartopy/shapely
    # bug (GEOSException: "Points of LinearRing do not form a closed
    # linestring") on some version combinations. Gridlines without labels
    # avoid the buggy path; tick labels are added manually instead.
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

    ax.set_xticks(range(-5, 40, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_lon))
    ax.tick_params(axis="x", labelsize=8)

    if show_lat_labels:
        ax.set_yticks(range(30, 47, 5), crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_format_lat))
        ax.tick_params(axis="y", labelsize=8)
    else:
        ax.set_yticks([])


def add_region_boxes(ax):
    """Draw PPCon's five evaluation sub-regions as labeled dashed boxes.

    Labels sit in a semi-transparent white box for legibility, since region
    boundaries often fall over open water or dense point clusters with no
    clean uniform background to place text against.
    """
    for name, (lat_range, lon_range) in REGIONS.items():
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range
        width = lon_max - lon_min
        height = lat_max - lat_min

        rect = mpatches.Rectangle(
            (lon_min, lat_min), width, height,
            linewidth=1.0,
            edgecolor=REGION_BOX_COLOR,
            facecolor="none",
            linestyle="--",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        ax.add_patch(rect)

        ax.text(
            lon_min + 0.3, lat_max - 1.0, name,
            fontsize=8, fontweight="bold", color=REGION_BOX_COLOR,
            transform=ccrs.PlateCarree(), zorder=5,
            bbox=dict(
                facecolor="white", edgecolor="none",
                alpha=0.75, pad=1.5, boxstyle="round,pad=0.2",
            ),
        )


def build_legend_handles():
    """Proxy artists for the shared colour legend.

    Plotted points are small and semi-transparent for density reasons; the
    legend swatches are drawn larger and fully opaque so the colour key is
    actually legible in print.
    """
    return [
        Line2D(
            [], [],
            marker="o",
            linestyle="none",
            markersize=LEGEND_MARKER_SIZE,
            markerfacecolor=VARIABLE_COLORS[variable],
            markeredgecolor="none",
            label=VARIABLE_LABELS[variable],
        )
        for variable in LEGEND_ORDER
    ]


def make_figure(seed=0):
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    data_by_split = load_all_splits()
    counts = {}

    for i, (ax, split) in enumerate(zip(axes, SPLITS)):
        add_basemap(ax, show_lat_labels=(i == 0))
        add_region_boxes(ax)

        for variable in VARIABLES:
            df = data_by_split[split][variable]
            jitter_lat = df["lat"] + rng.uniform(-JITTER_DEGREES, JITTER_DEGREES, size=len(df))
            jitter_lon = df["lon"] + rng.uniform(-JITTER_DEGREES, JITTER_DEGREES, size=len(df))

            ax.scatter(
                jitter_lon, jitter_lat,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                color=VARIABLE_COLORS[variable],
                transform=ccrs.PlateCarree(),
                edgecolors="none",
                zorder=2,
            )
            counts[(variable, split)] = len(df)

        ax.set_title(PANEL_TITLES[split], fontsize=12, fontweight="bold")

    # One legend only, on the train panel. Both panels use identical
    # colours, so repeating the key would be redundant. It sits in the
    # upper-right corner, over the Black Sea, which is inside the map
    # extent but holds no profiles and no sub-region boxes. The semi-
    # transparent white background matches the region labels.
    legend = axes[0].legend(
        handles=build_legend_handles(),
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        framealpha=0.75,
        edgecolor="none",
        facecolor="white",
        handletextpad=0.4,
        labelspacing=0.35,
        borderpad=0.5,
    )
    legend.set_zorder(6)

    return fig, counts


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, counts = make_figure()

    pdf_path = OUT_DIR / "spatial_coverage.pdf"
    png_path = OUT_DIR / "spatial_coverage.png"

    # PDF is vector for points/lines/text (only coastline polygons rasterize
    # internally), so it stays crisp at any zoom and is the file to drop
    # into Overleaf. PNG at 600 dpi is a high-res raster backup/preview.
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print()
    print("Profile counts:")
    print(f"{'Variable':<10} {'Train':>8} {'Test':>8}")
    for variable in VARIABLES:
        train_n = counts[(variable, "train")]
        test_n = counts[(variable, "test")]
        print(f"{variable:<10} {train_n:>8} {test_n:>8}")

    print()
    print("Suggested caption:")
    print(
        "Figure X. Spatial distribution of Mediterranean BGC-Argo profiles "
        "used in this study, inherited unmodified from PPCon's published "
        "train (left) and test (right) splits. A small random spatial "
        "jitter is applied so overlapping profiles remain visually "
        "distinguishable. Dashed boxes and labels indicate the five "
        "sub-regions used for regional evaluation in Results, following "
        "PPCon's own definitions: northwestern Mediterranean Sea (NWM), "
        "southwestern Mediterranean Sea (SWM), Tyrrhenian Sea (TYR), "
        "Ionian and southern Adriatic Sea (ION), and Levantine Sea (LEV)."
    )
    print()
    print("Overleaf: copy spatial_coverage.pdf into your figures/ folder and")
    print(r"reference with \includegraphics[width=\textwidth]{figures/spatial_coverage.pdf}")


if __name__ == "__main__":
    main()