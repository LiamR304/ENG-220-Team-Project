#!/usr/bin/env python
# coding: utf-8

# Imports

import io
import matplotlib
matplotlib.use("Agg")   # no GUI backend needed on a server

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
st.title("Cuba, New Mexico Broadband Coverage")
st.write("These maps and charts compare broadband coverage focusing on the availability of 5G mobile broadband and fixed broadband (\u2265 100/20 Mbps) in Cuba, NM and three other "
         "small New Mexico municipalities. Click any of the buttons down below to see the data and maps.")
st.divider()
# Data paths and constants:

in_dir = Path(__file__).parent / "Datasets"

hist_dir = in_dir / "Historical"

CURRENT_CSV = in_dir / "regional_h3_coord_bandwidth.csv"
HIST_CSV = {
    "2022": hist_dir / "cuba_h3_coord_bandwidth_2022.csv",
    "2025": hist_dir / "cuba_h3_coord_bandwidth_2025.csv",
}

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

colors = {
    "130077": {"35/3": "#08306b", "7/1": "#9ecae1"},   # blues
    "130403": {"35/3": "#e236a2", "7/1": "#f19cd1"},   # pinks
    "131425": {"35/3": "#db0909", "7/1": "#f85a5a"},   # reds
    "any":    {"35/3": "#333333", "7/1": "#bdbdbd"},   # grays
}

# Memory limits: the most hexagons one map will draw
MAX_MAP_CELLS = 20000            # one town
MAX_ALL_TOWNS_CELLS = 50000     # "All municipalities" view
ALL_TOWNS = "All"

# The same community bounding boxes Kenneth's original script used to build the regional CSV:
# (lat_min, lat_max, lon_min, lon_max)
COMMUNITY_BOXES = {
    "Cuba, NM":       (35.95, 36.05, -107.01, -106.9),
    "Chama, NM":      (36.85, 36.95, -106.6,  -106.5),
    "Taos, NM":       (36.2,  36.7,  -105.8,  -105.3),
    "Bloomfield, NM": (36.67, 36.79, -108.03, -107.91),
}

# The only columns the app ever reads (the CSVs have others, e.g. geometry_wkt, that are big)
NEEDED_COLS = ["h3_res9_id", "latitude", "longitude", "providerid", "mindown", "minup"]
# Cached loaders (Streamlit reruns the whole script on every click, so these keep it fast)
# cache_resource hands back the same object each time; cache_data would copy it on every call.

@st.cache_resource(max_entries=2)
def load_csv(path_str):
    # usecols skips the unused columns while reading, so they never take up memory
    d = pd.read_csv(path_str, usecols=lambda c: c.strip() in NEEDED_COLS)
    d.columns = d.columns.str.strip()
    d = d[NEEDED_COLS]
    d["h3_res9_id"] = d["h3_res9_id"].astype(str)
    d["latitude"] = d["latitude"].astype("float32")
    d["longitude"] = d["longitude"].astype("float32")
    d["providerid"] = d["providerid"].astype("int32")
    d["mindown"] = d["mindown"].astype("int16")
    d["minup"] = d["minup"].astype("int16")
    # One row per cell, provider, and tier is all any chart or map needs
    return d.drop_duplicates(subset=["h3_res9_id", "providerid", "mindown", "minup"]
                             ).reset_index(drop=True)


def load_boundary(path):
    # header=None + to_numeric handles files with or without a header row
    b = pd.read_csv(path, header=None, names=["lat", "lon"])
    b = b.apply(pd.to_numeric, errors="coerce").dropna()
    return list(zip(b["lat"], b["lon"]))


def cells_in_boundary(pts, res=9):
    poly = h3.LatLngPoly(pts)          # (lat, lon) pairs, no need to close the ring
    return set(h3.h3shape_to_cells(poly, res))


@st.cache_resource
def get_boundaries():
    return {name: load_boundary(p) for name, p in boundary_files.items()}


@st.cache_resource
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

    # All hexagons go in ONE GeoJSON layer. That is much lighter than one folium.Polygon
    # object per cell.
    features = []
    for h3_index in subset["h3_res9_id"]:
        ring = [[lng, lat] for lat, lng in h3.cell_to_boundary(h3_index)]
        ring.append(ring[0])
        features.append({
            "type": "Feature",
            "properties": {"h3": h3_index},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    style = {"color": color, "weight": 1, "fillColor": color, "fillOpacity": opacity}
    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name=title,
        style_function=lambda _feature: style,
        tooltip=folium.GeoJsonTooltip(fields=["h3"], aliases=["H3 cell"]),
    ).add_to(m)

    for name, (lat, lon) in towns.items():
        folium.Marker(
            location=[lat, lon],
            popup=name,
            tooltip=name,
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    # Town boundaries (outline only)
    boundary_layer = folium.FeatureGroup(name="Municipality boundaries")

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


# Coverage calculations (cached, and the big lookups are freed as soon as each one finishes)

def make_lookup(d):
    """{(providerid, mindown, minup): set of H3 ids}"""
    return {
        key: set(group["h3_res9_id"])
        for key, group in d.groupby(["providerid", "mindown", "minup"])
    }


def tier_coverage(lookup, total):
    """Yield (provider, tier, covered cells) for each provider and tier, plus 'any' provider."""
    for (mindown, minup) in speed_tiers:
        tier = f"{mindown}/{minup}"
        union_cells = set()
        for provider in providers:
            covered = lookup.get((provider, mindown, minup), set()) & total
            union_cells |= covered
            yield str(provider), tier, covered
        # Any provider at this tier (union, not a sum)
        yield "any", tier, union_cells


@st.cache_data(show_spinner="Computing coverage municipality...")
def coverage_summary():
    lookup = make_lookup(load_csv(str(CURRENT_CSV)))
    rows = []
    for town, total in get_town_cells().items():
        for provider, tier, covered in tier_coverage(lookup, total):
            rows.append({
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
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Computing Cuba trend...")
def coverage_trend():
    cuba_total = get_town_cells()["Cuba, NM"]   # H3 cells inside the Cuba boundary
    paths = {
        "2022": HIST_CSV["2022"],
        "2025": HIST_CSV["2025"],
        "2026": CURRENT_CSV,
    }
    rows = []
    # Scans all datasets to determine if any gaps in coverage exist for the given years.
    # A missing provider/tier/year has no cells in the lookup, so it comes out as 0%
    for period, path in paths.items():
        lookup = make_lookup(load_csv(str(path)))
        for provider, tier, covered in tier_coverage(lookup, cuba_total):
            rows.append({
                "period": period,
                "provider": provider,
                "tier": tier,
                "covered_cells": len(covered),
                "pct_covered": round(100 * len(covered) / len(cuba_total), 1) if cuba_total else 0.0,
            })
    return pd.DataFrame(rows)


# Charts are drawn once, saved as small PNG images, and cached.
# Showing an image is much lighter than keeping live matplotlib figures around.

def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner="Drawing bar chart...")
def bar_chart_png():
    summary = coverage_summary()
    order = [str(p) for p in providers] + ["any"]
    tiers = ["35/3", "7/1"]
    x = np.arange(len(order))
    width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)

    for ax, town in zip(axes.flat, boundaries):
        sub = summary[summary["town"] == town]

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

    fig.suptitle("Broadband coverage by municipality, provider, and speed tier", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig_to_png(fig)


@st.cache_data(show_spinner="Drawing line chart...")
def line_chart_png():
    trend = coverage_trend()
    periods = ["2022", "2025", "2026"]
    panel_titles = {"130077": "AT&T", "130403": "T-Mobile", "131425": "Verizon", "any": "Any provider"}
    order = [str(p) for p in providers] + ["any"]
    tiers = ["35/3", "7/1"]
    x = np.arange(len(periods))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)

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
    return fig_to_png(fig)




# ---- Locations without fixed broadband (FCC served/unserved table, pre-summarised per hexagon) ----
UNSERVED_HEX_CSV = Path("Datasets/unserved_by_hex.csv")
UNSERVED_TOWN_CSV = Path("Datasets/unserved_by_town.csv")
MAX_UNSERVED_CELLS = 20000
UNSERVED_MEASURES = {
    "Wired + licensed fixed wireless": "wiredlfw",
    "All terrestrial (wired + all fixed wireless)": "terrestrial",
    "Wired only (cable, fiber, copper)": "wired",
}
SERVICE_KINDS = {
    "Residential service": "res",
    "Business service": "bus",
    "Either (residential or business)": "all",
}
# (upper bound %, colour, label)
UNSERVED_BINS = [(0, "#c7e9c0", "0% (all served)"), (25, "#fdd0a2", ">0-25%"), (50, "#fdae6b", "26-50%"),
                 (75, "#e6550d", "51-75%"), (100.1, "#a63603", "76-100%")]


@st.cache_data
def load_unserved_hexes():
    return pd.read_csv(UNSERVED_HEX_CSV, dtype={"h3_res8_id": str})


@st.cache_data
def load_unserved_towns():
    return pd.read_csv(UNSERVED_TOWN_CSV)


def unserved_color(pct):
    for upper, color, _ in UNSERVED_BINS:
        if pct <= upper:
            return color
    return UNSERVED_BINS[-1][1]


def build_unserved_map(cells, key):
    pct_col, n_col = f"pct_unserved_{key}", f"unserved_{key}"
    m = folium.Map(tiles=None)
    folium.TileLayer(
        tiles=tiles_url,
        attr="\u00a9 CARTO \u00a9 OpenStreetMap contributors",
        name="CARTO Voyager",
    ).add_to(m)
    features = []
    for h, pct, n_un, tot in zip(cells["h3_res8_id"], cells[pct_col], cells[n_col], cells["total"]):
        ring = [[lng, lat] for lat, lng in h3.cell_to_boundary(h)]
        ring.append(ring[0])
        features.append({
            "type": "Feature",
            "properties": {"h3": h, "pct": f"{pct:.0f}%",
                           "detail": f"{int(n_un)} of {int(tot)} locations without service",
                           "fill": unserved_color(pct)},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="Locations without service",
        style_function=lambda f: {"color": f["properties"]["fill"], "weight": 0.5,
                                  "fillColor": f["properties"]["fill"], "fillOpacity": 0.75},
        tooltip=folium.GeoJsonTooltip(fields=["pct", "detail", "h3"],
                                      aliases=["Without service", "Locations", "H3 hexagon"]),
    ).add_to(m)
    m.fit_bounds([[cells["latitude"].min(), cells["longitude"].min()],
                  [cells["latitude"].max(), cells["longitude"].max()]])
    legend = "".join(
        f'<div><span style="background:{c};display:inline-block;width:14px;height:14px;'
        f'margin-right:6px;vertical-align:middle"></span>{lab}</div>' for _, c, lab in UNSERVED_BINS)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:30px;left:10px;z-index:9999;background:#ffffff;'
        'padding:8px 10px;border:1px solid #999;border-radius:4px;font-size:13px;'
        'color:#111;line-height:1.5">'
        f'<b>% of locations in hexagon<br>without service</b>{legend}</div>'))
    return m


# ---- Homes vs businesses (FCC place summary: share of units served, by town) ----
FIXED_SUMMARY_CSV = Path("Datasets/fixed_summary_by_town.csv")
TECH_CHOICES = {
    "Wired + fixed wireless, incl. unlicensed": "Any Terrestrial",
    "Wired + licensed fixed wireless": "All Wired and Licensed Fixed Wireless",
    "Wired only (cable, fiber, copper)": "All Wired",
}


@st.cache_data
def load_fixed_summary():
    return pd.read_csv(FIXED_SUMMARY_CSV)


@st.cache_data(show_spinner="Drawing chart...")
def homes_vs_business_png(tech):
    d = load_fixed_summary()
    d = d[d["technology"] == tech]
    town_order = [t for t in boundaries if t in set(d["town"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    x = np.arange(len(town_order))
    width = 0.38
    for ax, (col, label) in zip(axes, [("served_25_3", "25/3 Mbps"), ("served_100_20", "100/20 Mbps")]):
        for i, (kind, name, color) in enumerate([("R", "Residential service", "#4c78a8"), ("B", "Business service", "#f58518")]):
            vals = [100 - d[(d["town"] == t) & (d["biz_res"] == kind)][col].iloc[0] for t in town_order]
            bars = ax.bar(x + (i - 0.5) * width, vals, width, color=color, edgecolor="black",
                          linewidth=0.5, label=name)
            ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)
        ax.set_title(f"No {label} service", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace(", NM", "") for t in town_order])
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("% of all units")
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Share of units without residential or business fixed broadband", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig_to_png(fig)


# Page layout: only the section picked is computed, so the app never holds all three at once

section = st.radio(
    "Section",
    ["About & Resources", "5G Mobile Broadband Coverage Map", "5G Broadband Coverage by Municipality", "Cuba 5G Coverage Over Time", "Locations Without Fixed Broadband", "Residential vs Business Service"],
    horizontal=True,
    label_visibility="collapsed",
)


if section == "About & Resources":
    st.header("About This Dashboard")
    st.write(
        "This dashboard compares 5G and fixed broadband coverage in four New Mexico "
        "communities: Cuba, Chama, Taos and Bloomfield. It was built for ENG 220 (Peace "
        "Engineering) to show where service is strong, where it is thin, and how Cuba "
        "compares with its neighbors. We hope in particular that the residents of Cuba, NM may find it of some use."
    )
    st.subheader("What the Sections Show")
    st.markdown(
        "- **5G Mobile Broadband Coverage map:** where each provider reports 5G coverage, by speed tier, as H3 hexagons.\n"
        "- **5G Broadband Coverage by Municipality:** the share of each town's area covered, by provider and speed tier.\n"
        "- **Cuba 5G Coverage Over Time:** how Cuba's coverage changed between 2022, 2025 and 2026.\n"
        "- **Locations without fixed broadband:** the share of locations in each "
        "hexagon that no provider reports serving at 100 Mbps down / 20 Mbps up, or the share served.\n"
        "- **Residential vs Business Service:** for each town, the share of units where providers report residential-grade fixed broadband compared with business-grade fixed broadband."
    )
    st.subheader("Where the Data Comes From")
    st.write(
        "All figures come from the FCC. Providers report their own "
        "coverage, so a place shown as covered may still have weak or no service in practice."
    )
    st.subheader("Project Limitations")
    st.markdown(
        "- Coverage is self-reported by providers and is not a measured speed test.\n"
        "- On the fixed-broadband map, \"residential\" and \"business\" mean the type of customer a "
        "provider says it offers service to at a location, not the type of building.\n"
        "- Hexagon percentages are noisy where a hexagon has only a few locations; use the "
        "minimum-locations slider.\n"
        "-Exact number of locations/units per hexagon is likely not exact.\n"
        "- The fixed-broadband section uses the FCC data release for June 2026 (last updated November 10, 2026)."
        
    )

    st.header("Resources for Residents")
    st.markdown(
        "**Check or correct your address:**\n"
        "- [FCC National Broadband Map](https://broadbandmap.fcc.gov): look up the providers "
        "reported at your address.\n"
        "- [How to submit a location challenge](https://help.bdc.fcc.gov/hc/en-us/articles/10475216120475-How-to-Submit-a-Location-Challenge): "
        "if your home or business is missing or listed wrongly.\n"
        "- [How to file an availability challenge](https://help.bdc.fcc.gov/hc/en-us/articles/10476040597787-How-to-File-an-Availability-Challenge): "
        "if a provider is listed but does not actually serve you.\n\n"
        "**Help paying for service:**\n"
        "- [FCC Lifeline program](https://www.fcc.gov/lifeline-consumer): a monthly discount on "
        "phone or internet service for eligible low-income households.\n\n"
        "**State of New Mexico programs:**\n"
        "- [Connect New Mexico](https://connect.nm.gov/digital-opportunity.html): the state "
        "Office of Broadband Access and Expansion, with digital-opportunity resources.\n"
        "- [More info on resources available and programs in New Mexico](https://connect.nm.gov/programs.html): state "
        "information on digital needs and how communities can access resources."
    )


elif section == "5G Mobile Broadband Coverage Map":
    st.header("5G Mobile Broadband Coverage Map")

    st.sidebar.header("Map Options")   # optional title for the sidebar
    dataset = st.sidebar.selectbox("Dataset", ["2026 (all municipalities)", "Cuba 2025", "Cuba 2022"])

    if dataset == "2026 (all municipalities)":
        area = st.sidebar.selectbox("Municipality", [ALL_TOWNS] + list(boundaries))
        map_df = load_csv(str(CURRENT_CSV))
    else:
        area = "Cuba, NM"
        st.sidebar.selectbox("Municipality", [area], disabled=True)
        map_df = load_csv(str(HIST_CSV[dataset.split()[-1]]))

    provider = st.sidebar.selectbox("Provider", providers, format_func=lambda p: provider_names[str(p)])
    tier_sel = st.sidebar.selectbox("Speed tier (down/up Mbps)", list(speed_tiers),
                                    format_func=lambda t: f"{t[0]}/{t[1]}")

    mindown, minup = tier_sel
    style = speed_tiers[tier_sel]

    # Draw the cells inside the chosen town's community box (or all four boxes)
    selected = list(boundaries) if area == ALL_TOWNS else [area]
    cap = MAX_ALL_TOWNS_CELLS if area == ALL_TOWNS else MAX_MAP_CELLS

    tier_df = map_df[
        (map_df["providerid"] == provider)
        & (map_df["mindown"] == mindown)
        & (map_df["minup"] == minup)
    ]

    # A cell is kept if it falls in the community box of ANY selected town
    in_area = pd.Series(False, index=tier_df.index)
    for name in selected:
        lat_lo, lat_hi, lon_lo, lon_hi = COMMUNITY_BOXES[name]
        in_area |= (
            tier_df["latitude"].between(lat_lo, lat_hi)
            & tier_df["longitude"].between(lon_lo, lon_hi)
        )
    subset = tier_df[in_area]

    if subset.empty:
        st.warning("No coverage data for this municipality, provider, speed tier, and dataset. (From period before widescale 5G adoption)")
    else:
        total_cells = len(subset)
        if total_cells > cap:
            st.info(f"{total_cells:,} cells here. To keep the app fast, the map shows a random "
                    f"sample of {cap:,}. The charts use every cell.")
            subset = subset.sample(cap, random_state=0)
        where = "across all four towns" if area == ALL_TOWNS else f"near {area}"
        st.write(f"{total_cells:,} H3 cells (about {total_cells * CELL_KM2:,.1f} km²) with coverage "
                 f"{where}.")
        color = provider_colors[provider][style["shade"]]
        title = f"{dataset} - {area} - {provider_names[str(provider)]} - {mindown}/{minup} Mbps"
        m = build_map(subset, color, style["opacity"], title,
                      boundaries={n: boundaries[n] for n in selected},
                      towns={n: towns[n] for n in selected})
        st_folium(m, height=600, use_container_width=True, returned_objects=[])


elif section == "5G Broadband Coverage by Municipality":
    st.header("5G Broadband Coverage by Municipality")

    st.image(bar_chart_png(), use_container_width=True)

    with st.expander("Show data table (% covered)"):
        st.dataframe(coverage_summary().pivot_table(
            index=["town", "provider"],
            columns="tier",
            values="pct_covered",
        ))


elif section == "Cuba 5G Coverage Over Time":
    st.header("Cuba 5G Coverage Over Time")

    st.image(line_chart_png(), use_container_width=True)

    with st.expander("Show data table (% covered)"):
        st.dataframe(coverage_trend().pivot_table(
            index=["provider", "tier"], columns="period", values="pct_covered",
        )[["2022", "2025", "2026"]])


elif section == "Locations Without Fixed Broadband":
    st.header("Locations Without Fixed Broadband")
    st.caption("All fixed-broadband serviceable locations for Cuba, "
               "Chama, Taos and Bloomfield. A location counts as served if a provider reports 100 Mbps "
               "down / 20 Mbps up of the chosen type, offered to residential customers, business "
               "customers, or either. Percentages are out of all locations in the hexagon. Each hexagon is about 0.7 km\u00b2 (H3 resolution 8).")

    st.sidebar.header("Map Options")
    measure = st.sidebar.selectbox("Service type", list(UNSERVED_MEASURES))
    kind_label = st.sidebar.selectbox("Service offered to", list(SERVICE_KINDS))
    key = f"{UNSERVED_MEASURES[measure]}_{SERVICE_KINDS[kind_label]}"
    towns_all = load_unserved_towns()["town"].tolist()
    town_sel = st.sidebar.selectbox("Municipality", ["All municipalities"] + towns_all)
    min_locs = st.sidebar.slider("Minimum locations per hexagon", 1, 20, 1,
                                 help="Hexagons with very few locations give noisy percentages.")
    min_pct = st.sidebar.slider("Only show hexagons with at least this % unserved", 0, 100, 0)

    hexes = load_unserved_hexes()
    if town_sel != "All municipalities":
        hexes = hexes[hexes["town"] == town_sel]
    else:
        # a hexagon on a town edge can be listed for two towns; merge so it is drawn once
        sums = [c for c in hexes.columns if c == "total" or c.startswith("unserved_")]
        hexes = hexes.groupby("h3_res8_id", as_index=False).agg(
            {**{c: "sum" for c in sums}, "latitude": "first", "longitude": "first"})
        for c in [c for c in sums if c.startswith("unserved_")]:
            hexes["pct_" + c] = (100 * hexes[c] / hexes["total"]).round(1)
    cells = hexes[(hexes["total"] >= min_locs) & (hexes[f"pct_unserved_{key}"] >= min_pct)]
    total_un = int(hexes[f"unserved_{key}"].sum())
    st.write(f"{total_un:,} of {int(hexes['total'].sum()):,} locations "
             f"{'across the four towns' if town_sel == 'All municipalities' else 'in ' + town_sel} "
             f"({100 * total_un / hexes['total'].sum():.1f}%) lack this service. "
             f"{len(cells):,} hexagons match the filters.")

    if cells.empty:
        st.warning("No hexagons match these filters.")
    else:
        if len(cells) > MAX_UNSERVED_CELLS:
            st.info(f"To keep the app fast, the map shows the {MAX_UNSERVED_CELLS:,} hexagons "
                    f"with the most locations without service.")
            cells = cells.nlargest(MAX_UNSERVED_CELLS, f"unserved_{key}")
        st_folium(build_unserved_map(cells, key), height=600, use_container_width=True,
                  returned_objects=[])

    with st.expander("Show data table by municipality"):
        st.dataframe(load_unserved_towns().sort_values(f"unserved_{key}", ascending=False)[
            ["town", "total", f"unserved_{key}", f"pct_unserved_{key}"]
        ].rename(columns={"total": "locations", f"unserved_{key}": "without service",
                          f"pct_unserved_{key}": "% without service"}), hide_index=True)


elif section == "Residential vs Business Service":
    st.header("Residential vs Business Service")
    st.caption("From the FCC place summary (June 2026 release). Each bar is the share of all units in "
               "the town where no provider reports residential (or business) fixed broadband service at "
               "the speed shown. Residential and business service are reported separately, so a unit "
               "can have one, both or neither. Below 25/3 Mbps is generally treated as unserved; below "
               "100/20 as underserved or unserved.")
    tech_label = st.selectbox("Service type", list(TECH_CHOICES))
    tech = TECH_CHOICES[tech_label]
    st.image(homes_vs_business_png(tech), use_container_width=True)

    with st.expander("Show data table (% of all units with service reported)"):
        t = load_fixed_summary()
        t = t[t["technology"] == tech].copy()
        t["type"] = t["biz_res"].map({"R": "Residential service", "B": "Business service"})
        st.dataframe(t[["town", "type", "total_units", "served_25_3", "served_100_20"]].rename(columns={
            "total_units": "total units in town", "served_25_3": "% served 25/3",
            "served_100_20": "% served 100/20"}), hide_index=True)
