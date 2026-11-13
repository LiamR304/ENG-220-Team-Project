#!/usr/bin/env python
# coding: utf-8

# Imports

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import h3
import folium
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="NM Broadband Coverage", layout="wide")
st.title("New Mexico 5G Broadband Coverage")


# Data paths and constants:

in_dir = Path(__file__).parent / "Datasets"

hist_dir = in_dir / "Historical"

providers = [130077, 130403, 131425] #AT&T, T-Mobile, Verizon

boundary_files = {
    "Cuba, NM":       in_dir / "Boundaries" / "cuba_boundary.csv",
    "Chama, NM":      in_dir / "Boundaries" / "chama_boundary.csv",
    "Taos, NM":       in_dir / "Boundaries" / "taos_boundary.csv",
    "Bloomfield, NM": in_dir / "Boundaries" / "bloomfield_boundary.csv",
}

RES = 9
CELL_KM2 = h3.average_hexagon_area(RES, unit="km^2")   # ~0.105 km2 (h3 v4)
CELL_SQMI = CELL_KM2 * 0.386102

provider_colors = {
    130077: {"dark": "#08306b", "light": "#9ecae1"},   # blues
    130403: {"dark": "#e236a2", "light": "#f19cd1"},   # pinks
    131425: {"dark": "#db0909", "light": "#f85a5a"},   # reds
}

speed_tiers = {
    (35, 3): {"shade": "dark",  "opacity": 0.65},
    (7, 1):  {"shade": "light", "opacity": 0.55},
}

towns = {
    "Cuba, NM":       (36.0222, -106.9584),
    "Chama, NM":      (36.903,  -106.579),
    "Taos, NM":       (36.4072, -105.5731),
    "Bloomfield, NM": (36.7400, -107.9845),
}

provider_names = {
    "130077": "AT&T",
    "130403": "T-Mobile",
    "131425": "Verizon",
    "any": "Any\nprovider",
}


# Cached loaders (Streamlit reruns the whole script on every click, so these keep it fast)

@st.cache_data
def load_csv(path_str):
    d = pd.read_csv(path_str)
    d.columns = d.columns.str.strip()
    d["h3_res9_id"] = d["h3_res9_id"].astype(str)
    return d


def load_boundary(path):
    # header=None + to_numeric handles files with or without a header row
    b = pd.read_csv(path, header=None, names=["lat", "lon"])
    b = b.apply(pd.to_numeric, errors="coerce").dropna()
    return list(zip(b["lat"], b["lon"]))


def cells_in_boundary(pts, res=9):
    poly = h3.LatLngPoly(pts)          # (lat, lon) pairs, no need to close the ring
    return set(h3.h3shape_to_cells(poly, res))


@st.cache_data
def get_boundaries():
    return {name: load_boundary(p) for name, p in boundary_files.items()}


@st.cache_data
def get_town_cells():
    return {name: cells_in_boundary(pts) for name, pts in get_boundaries().items()}


boundaries = get_boundaries()

# Historical maps show Cuba only, so overlay just the Cuba boundary and town marker
cuba_boundaries = {"Cuba, NM": boundaries["Cuba, NM"]}
cuba_towns      = {"Cuba, NM": towns["Cuba, NM"]}


# Build the map

tiles_url = (
    "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_50ry_1_cccdef19d6d2d84e86a24f0b"
)


def build_map(subset, color, opacity, title, boundaries=boundaries, towns=towns):

    b_lats = [p[0] for pts in boundaries.values() for p in pts]
    b_lons = [p[1] for pts in boundaries.values() for p in pts]

    lats = list(subset["latitude"]) + b_lats
    lons = list(subset["longitude"]) + b_lons

    m = folium.Map(tiles=None)
    folium.TileLayer(
        tiles=tiles_url,
        attr="© CARTO © OpenStreetMap contributors",
        name="CARTO Voyager",
    ).add_to(m)

    hex_layer = folium.FeatureGroup(name=title)
    for row in subset.itertuples(index=False):
        h3_index = str(row.h3_res9_id)
        folium.Polygon(
            locations=h3.cell_to_boundary(h3_index),
            color=color,            # outline
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            tooltip=(f"H3: {h3_index}<br>"
                     f"Provider: {row.providerid}<br>"
                     f"Down/Up: {row.mindown}/{row.minup} Mbps"),
        ).add_to(hex_layer)
    hex_layer.add_to(m)

    for name, (lat, lon) in towns.items():
        folium.Marker(
            location=[lat, lon],
            popup=name,
            tooltip=name,
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    # Town boundaries (outline only)
    boundary_layer = folium.FeatureGroup(name="Town boundaries")

    for name, pts in boundaries.items():
        folium.Polygon(
            locations=pts,
            color="black",
            weight=2.5,
            dash_array="6, 4",
            fill=False,
            tooltip=f"{name} boundary",
        ).add_to(boundary_layer)
    boundary_layer.add_to(m)

    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    folium.LayerControl().add_to(m)
    return m


# Coverage map: pick a dataset, provider, and speed tier

st.header("Coverage map")

c1, c2, c3 = st.columns(3)
dataset = c1.selectbox("Dataset", ["2026 (all towns)", "Cuba 2025", "Cuba 2022"])
provider = c2.selectbox("Provider", providers, format_func=lambda p: provider_names[str(p)])
tier_sel = c3.selectbox("Speed tier (down/up Mbps)", list(speed_tiers),
                        format_func=lambda t: f"{t[0]}/{t[1]}")
mindown, minup = tier_sel
style = speed_tiers[tier_sel]

if dataset == "2026 (all towns)":
    map_df = load_csv(str(in_dir / "regional_h3_coord_bandwidth.csv"))
    map_boundaries, map_towns = boundaries, towns
else:
    year = dataset.split()[-1]
    map_df = load_csv(str(hist_dir / f"cuba_h3_coord_bandwidth_{year}.csv"))
    map_boundaries, map_towns = cuba_boundaries, cuba_towns

subset = map_df[
    (map_df["providerid"] == provider)
    & (map_df["mindown"] == mindown)
    & (map_df["minup"] == minup)
].drop_duplicates(subset="h3_res9_id")

if subset.empty:
    st.warning("No coverage data for this provider, speed tier, and dataset.")
else:
    st.write(f"{len(subset):,} H3 cells (about {len(subset) * CELL_KM2:,.1f} km²) with coverage.")
    color = provider_colors[provider][style["shade"]]
    title = f"{dataset} - {provider_names[str(provider)]} - {mindown}/{minup} Mbps"
    m = build_map(subset, color, style["opacity"], title,
                  boundaries=map_boundaries, towns=map_towns)
    st_folium(m, height=600, returned_objects=[])


# Determine coverage and plot

town_cells = get_town_cells()

# -----------------------------------
# 3. Load coverage data and index it once
# -----------------------------------
df = load_csv(str(in_dir / "regional_h3_coord_bandwidth.csv"))

# Unique covered cells for each (provider, mindown, minup)
covered_lookup = {
    key: set(group["h3_res9_id"])
    for key, group in df.groupby(["providerid", "mindown", "minup"])
}

# -----------------------------------
# 4. Compute coverage per town, provider, and tier
# -----------------------------------
def make_row(town, provider, tier, covered, total):
    return {
        "town": town,
        "provider": provider,
        "tier": tier,
        "covered_cells": len(covered),
        "total_cells": len(total),
        "covered_km2": round(len(covered) * CELL_KM2, 2),
        "total_km2": round(len(total) * CELL_KM2, 2),
        "covered_sqmi": round(len(covered) * CELL_SQMI, 2),
        "total_sqmi": round(len(total) * CELL_SQMI, 2),
        "pct_covered": round(100 * len(covered) / len(total), 1) if total else None,
    }


rows = []

for town, total in town_cells.items():
    for (mindown, minup) in speed_tiers:
        tier = f"{mindown}/{minup}"
        union_cells = set()

        for provider in providers:
            cells = covered_lookup.get((provider, mindown, minup), set())
            covered = cells & total          # only cells inside the town
            union_cells |= covered
            rows.append(make_row(town, provider, tier, covered, total))

        # Any provider at this tier (union, not a sum)
        rows.append(make_row(town, "any", tier, union_cells, total))

summary = pd.DataFrame(rows)

# -----------------------------------
# 5. Display the summary table
# -----------------------------------
st.header("Coverage by town")

with st.expander("Show data table (% covered)"):
    st.dataframe(summary.astype({"provider": str}).pivot_table(
        index=["town", "provider"],
        columns="tier",
        values="pct_covered",
    ))

# -----------------------------------
# 6. Bar chart
# -----------------------------------
colors = {
    "130077": {"35/3": "#08306b", "7/1": "#9ecae1"},   # blues
    "130403": {"35/3": "#e236a2", "7/1": "#f19cd1"},   # pinks
    "131425": {"35/3": "#db0909", "7/1": "#f85a5a"},   # reds
    "any":    {"35/3": "#333333", "7/1": "#bdbdbd"},   # grays
}

plot_df = summary.copy()
plot_df["provider"] = plot_df["provider"].astype(str)

order = [str(p) for p in providers] + ["any"]
tiers = ["35/3", "7/1"]
x = np.arange(len(order))
width = 0.38

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)

for ax, town in zip(axes.flat, town_cells):
    sub = plot_df[plot_df["town"] == town]

    for i, tier in enumerate(tiers):
        vals = [
            sub[(sub["provider"] == p) & (sub["tier"] == tier)]["pct_covered"].iloc[0]
            for p in order
        ]
        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            color=[colors[p][tier] for p in order],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)

    ax.set_title(town, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([provider_names[p] for p in order])
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of town area covered")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

# Legend explains shade = speed tier (provider is identified by the x-axis)
fig.legend(
    handles=[
        Patch(facecolor="#444444", edgecolor="black", label="35/3 Mbps (dark)"),
        Patch(facecolor="#cccccc", edgecolor="black", label="7/1 Mbps (light)"),
    ],
    loc="lower center", ncol=2, frameon=False, fontsize=10,
)

fig.suptitle("Broadband coverage by town, provider, and speed tier", fontsize=14)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])

st.pyplot(fig)
plt.close(fig)


# Line chart showing coverage trend over data from 2022, 2025, and 2026:

st.header("Cuba coverage over time")

cuba_total = town_cells["Cuba, NM"]          # H3 cells inside the Cuba boundary

def build_lookup(df):
    """{(providerid, mindown, minup): set of H3 ids}, same shape as covered_lookup."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["h3_res9_id"] = df["h3_res9_id"].astype(str)
    return {
        key: set(group["h3_res9_id"])
        for key, group in df.groupby(["providerid", "mindown", "minup"])
    }

# Scans all datasets to determine if any gaps in coverage exist for the given years.
# A missing provider/tier/year has no cells in the lookup, so it comes out as 0%
year_lookups = {
    "2022": build_lookup(load_csv(str(hist_dir / "cuba_h3_coord_bandwidth_2022.csv"))),
    "2025": build_lookup(load_csv(str(hist_dir / "cuba_h3_coord_bandwidth_2025.csv"))),
    "2026": covered_lookup,
}
periods = list(year_lookups)

def pct(covered):
    return round(100 * len(covered) / len(cuba_total), 1) if cuba_total else 0.0

trend_rows = []
for period, lookup in year_lookups.items():
    for (mindown, minup) in [(35, 3), (7, 1)]:
        tier = f"{mindown}/{minup}"
        union_cells = set()
        for provider in providers:
            covered = lookup.get((provider, mindown, minup), set()) & cuba_total
            union_cells |= covered
            trend_rows.append({"period": period, "provider": str(provider), "tier": tier,
                               "covered_cells": len(covered), "pct_covered": pct(covered)})
        trend_rows.append({"period": period, "provider": "any", "tier": tier,
                           "covered_cells": len(union_cells), "pct_covered": pct(union_cells)})

trend = pd.DataFrame(trend_rows)

with st.expander("Show data table (% covered)"):
    st.dataframe(trend.pivot_table(index=["provider", "tier"], columns="period",
                                   values="pct_covered")[periods])

#Set up the line chart
panel_titles = {"130077": "AT&T", "130403": "T-Mobile", "131425": "Verizon", "any": "Any provider"}
order = [str(p) for p in providers] + ["any"]
tiers = ["35/3", "7/1"]
x = np.arange(len(periods))

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)

for ax, p in zip(axes.flat, order):
    vals = {
        tier: [
            trend[(trend["provider"] == p) & (trend["tier"] == tier) & (trend["period"] == per)]
            ["pct_covered"].iloc[0]
            for per in periods
        ]
        for tier in tiers
    }

    for tier in tiers:
        ax.plot(x, vals[tier], color=colors[p][tier], marker="o", markersize=8, linewidth=2.5,
                markeredgecolor="black", markeredgewidth=0.5)

    # Label the higher point of each pair above and the lower one below so labels never collide
    for i, xi in enumerate(x):
        v35, v7 = vals["35/3"][i], vals["7/1"][i]
        for tier, v in (("35/3", v35), ("7/1", v7)):
            other = v7 if tier == "35/3" else v35
            above = v > other or (v == other and tier == "7/1")
            ax.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points",
                        xytext=(0, 7 if above else -14), ha="center", fontsize=8)

    ax.set_title(panel_titles[p], fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.set_xlim(-0.3, len(periods) - 0.7)
    ax.set_ylim(-14, 112)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of Cuba area covered")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

fig.legend(
    handles=[
        Line2D([0], [0], color="#444444", marker="o", markeredgecolor="black", label="35/3 Mbps (dark)"),
        Line2D([0], [0], color="#cccccc", marker="o", markeredgecolor="black", label="7/1 Mbps (light)"),
    ],
    loc="lower center", ncol=2, frameon=False, fontsize=10,
)

fig.suptitle("Cuba, NM broadband coverage over time, by provider and speed tier", fontsize=14)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])

st.pyplot(fig)
plt.close(fig)