#!/usr/bin/env python
# coding: utf-8

# Imports

# In[28]:


import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import h3
import folium
from folium.plugins import MarkerCluster
import html
import streamlit as st


# Data paths and constants:

# In[20]:


in_dir = Path("F:/Anaconda/envs/ENG220/Datasets")

geopacks = [
    in_dir / "NM_ATT_5GNR_35_3_mobile_broadband.gpkg",
    in_dir / "NM_ATT_5GNR_7_1_mobile_broadband.gpkg",
    in_dir / "NM_TMOB_5GNR_35_3_mobile_broadband.gpkg",
    in_dir / "NM_TMOB_5GNR_7_1_mobile_broadband.gpkg",
    in_dir / "NM_VERI_5GNR_35_3_mobile_broadband.gpkg",
    in_dir / "NM_VERI_5GNR_7_1_mobile_broadband.gpkg",
]

hist_dir = in_dir / "Historical"

geopacks_his_2025 = [
    hist_dir / "NM_ATT_5GNR_35_3_mobile_broadband_2025.gpkg",
    hist_dir / "NM_ATT_5GNR_7_1_mobile_broadband_2025.gpkg",
    hist_dir / "NM_TMOB_5GNR_35_3_mobile_broadband_2025.gpkg",
    hist_dir / "NM_TMOB_5GNR_7_1_mobile_broadband_2025.gpkg",
    hist_dir / "NM_VERI_5GNR_35_3_mobile_broadband_2025.gpkg",
    hist_dir / "NM_VERI_5GNR_7_1_mobile_broadband_2025.gpkg",
]

geopacks_his_2022 = [
    hist_dir / "NM_ATT_5GNR_35_3_mobile_broadband_2022.gpkg",
    hist_dir / "NM_ATT_5GNR_7_1_mobile_broadband_2022.gpkg",
    hist_dir / "NM_TMOB_5GNR_35_3_mobile_broadband_2022.gpkg",
    hist_dir / "NM_TMOB_5GNR_7_1_mobile_broadband_2022.gpkg",
    hist_dir / "NM_VERI_5GNR_35_3_mobile_broadband_2022.gpkg",
    hist_dir / "NM_VERI_5GNR_7_1_mobile_broadband_2022.gpkg",
]

gpkg_stems = [
    "NM_ATT_5GNR_35_3_mobile_broadband",
    "NM_ATT_5GNR_7_1_mobile_broadband",
    "NM_TMOB_5GNR_35_3_mobile_broadband",
    "NM_TMOB_5GNR_7_1_mobile_broadband",
    "NM_VERI_5GNR_35_3_mobile_broadband",
    "NM_VERI_5GNR_7_1_mobile_broadband",
]

gpkg_stems_2025 = [
    "NM_ATT_5GNR_35_3_mobile_broadband_2025",
    "NM_ATT_5GNR_7_1_mobile_broadband_2025",
    "NM_TMOB_5GNR_35_3_mobile_broadband_2025",
    "NM_TMOB_5GNR_7_1_mobile_broadband_2025",
    "NM_VERI_5GNR_35_3_mobile_broadband_2025",
    "NM_VERI_5GNR_7_1_mobile_broadband_2025",
]

gpkg_stems_2022 = [
    "NM_ATT_5GNR_35_3_mobile_broadband_2022",
    "NM_ATT_5GNR_7_1_mobile_broadband_2022",
    "NM_TMOB_5GNR_35_3_mobile_broadband_2022",
    "NM_TMOB_5GNR_7_1_mobile_broadband_2022",
    "NM_VERI_5GNR_35_3_mobile_broadband_2022",
    "NM_VERI_5GNR_7_1_mobile_broadband_2022",
]

dec_dir = in_dir / "decoded"
dec_dir.mkdir(exist_ok=True)

out_dir = in_dir / "Sectional Datasets"
out_dir.mkdir(exist_ok=True)

map_dir = in_dir / "Maps"
map_dir.mkdir(exist_ok=True)

communities = {
    "cuba":  (*sorted([35.95, 36.05]), *sorted([-107.01, -106.9])),
    "chama": (*sorted([36.85, 36.95]), *sorted([-106.6, -106.5])),
    "taos":  (*sorted([36.2, 36.7]),   *sorted([-105.8, -105.3])),
    "bloom": (*sorted([36.67, 36.79]), *sorted([-108.03, -107.91])),
}

providers = [130077, 130403, 131425] #AT&T, T-Mobile, Verizon
speed_tiers = [(35, 3), (7, 1)]  #Download, Upload

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


# (GEOPACKS REQUIRED) Load and sort data into their providers and speeds

# In[ ]:


for gpkg in geopacks:
    gdf = gpd.read_file(gpkg)
    gdf["geometry_wkt"] = gdf.geometry.to_wkt()
    df_to_export = gdf.drop(columns="geometry")

    out_path = in_dir / f"{gpkg.stem}.csv"   # unique name per input
    df_to_export.to_csv(out_path, index=False)
    print(f"{gpkg.name} -> {out_path.name} ({len(df_to_export)} rows)")
#print(gdf.head())

csv_files = [in_dir / f"{stem}.csv" for stem in gpkg_stems]

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Decode each H3 cell once, then split into two columns
    centers = df["h3_res9_id"].astype(str).map(h3.cell_to_latlng)
    df["latitude"] = centers.str[0]
    df["longitude"] = centers.str[1]

    out_path = dec_dir / f"{csv_path.stem}_decoded.csv"
    df.to_csv(out_path, index=False)
    print(f"{csv_path.name} -> {out_path.name} ({len(df)} rows)")


# (GEOPACKS REQUIRED) Load and sort historical data into their providers and speeds

# In[ ]:


for gpkg in geopacks_his_2025:
    gdf = gpd.read_file(gpkg)
    gdf["geometry_wkt"] = gdf.geometry.to_wkt()
    df_to_export = gdf.drop(columns="geometry")

    out_path = hist_dir / f"{gpkg.stem}.csv"   # unique name per input
    df_to_export.to_csv(out_path, index=False)
    print(f"{gpkg.name} -> {out_path.name} ({len(df_to_export)} rows)")
#print(gdf.head())

csv_files = [hist_dir / f"{stem}.csv" for stem in gpkg_stems_2025]

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Decode each H3 cell once, then split into two columns
    centers = df["h3_res9_id"].astype(str).map(h3.cell_to_latlng)
    df["latitude"] = centers.str[0]
    df["longitude"] = centers.str[1]

    out_path = dec_dir / f"{csv_path.stem}_decoded.csv"
    df.to_csv(out_path, index=False)
    print(f"{csv_path.name} -> {out_path.name} ({len(df)} rows)")

for gpkg in geopacks_his_2022:
    gdf = gpd.read_file(gpkg)
    gdf["geometry_wkt"] = gdf.geometry.to_wkt()
    df_to_export = gdf.drop(columns="geometry")

    out_path = hist_dir / f"{gpkg.stem}.csv"   # unique name per input
    df_to_export.to_csv(out_path, index=False)
    print(f"{gpkg.name} -> {out_path.name} ({len(df_to_export)} rows)")
#print(gdf.head())

csv_files = [hist_dir / f"{stem}.csv" for stem in gpkg_stems_2022]

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Decode each H3 cell once, then split into two columns
    centers = df["h3_res9_id"].astype(str).map(h3.cell_to_latlng)
    df["latitude"] = centers.str[0]
    df["longitude"] = centers.str[1]

    out_path = dec_dir / f"{csv_path.stem}_decoded.csv"
    df.to_csv(out_path, index=False)
    print(f"{csv_path.name} -> {out_path.name} ({len(df)} rows)")


# (GEOPACKS REQUIRED) Filter data to the boundings of each community, then combine and save merged filtered dataset.

# In[ ]:

decoded_files = [dec_dir / f"{stem}_decoded.csv" for stem in gpkg_stems]
all_regions = []
for csv_path in decoded_files:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    source = csv_path.stem.replace("_decoded", "")

    for name, (lat_min, lat_max, lon_min, lon_max) in communities.items():
        region = df[
            df["latitude"].between(lat_min, lat_max) &
            df["longitude"].between(lon_min, lon_max)
        ].copy()

        if region.empty:
            print(f"{source}: no cells in {name}")
            continue

        region["community"] = name
        region["source_file"] = source

        # One CSV per source file and community
        region.to_csv(out_dir / f"{name}_{source}_h3_cells.csv", index=False)
        all_regions.append(region)
        print(f"{source}: {len(region)} cells in {name}")

# Combine everything into one dataset (for the folium map)
if all_regions:
    df_combined = pd.concat(all_regions, ignore_index=True)
    df_combined.to_csv(in_dir / "regional_h3_coord_bandwidth.csv", index=False)
    print(f"Combined: {len(df_combined)} rows")
else:
    print("No cells matched any community. Check your bounding boxes.")


# (GEOPACKS REQUIRED) Filter, merge, and save provider coverage datasets for just Cuba in years 2025 and 2022

# In[ ]:


# Cuba bounding box: (lat_min, lat_max, lon_min, lon_max)
cuba_lat_min, cuba_lat_max, cuba_lon_min, cuba_lon_max = communities["cuba"]

hist_years = {
    2025: gpkg_stems_2025,
    2022: gpkg_stems_2022,
}

required_cols = {"h3_res9_id", "latitude", "longitude", "providerid", "mindown", "minup"}
cuba_hist = {}   # year -> filtered DataFrame

for year, stems in hist_years.items():
    frames = []

    for stem in stems:
        csv_path = dec_dir / f"{stem}_decoded.csv"
        df_h = pd.read_csv(csv_path)
        df_h.columns = df_h.columns.str.strip()

        #Checks if historical datasets are formatted correctly
        missing = required_cols - set(df_h.columns)
        if missing:
            raise KeyError(f"{csv_path.name} is missing columns: {sorted(missing)}")

        cuba = df_h[
            df_h["latitude"].between(cuba_lat_min, cuba_lat_max) &
            df_h["longitude"].between(cuba_lon_min, cuba_lon_max)
        ].copy()

        cuba["community"] = "cuba"
        cuba["source_file"] = stem
        cuba["year"] = year
        frames.append(cuba)
        print(f"{year} | {stem}: {len(cuba)} cells in cuba")

    df_year = pd.concat(frames, ignore_index=True)
    cuba_hist[year] = df_year

    out_path = hist_dir / f"cuba_h3_coord_bandwidth_{year}.csv"
    df_year.to_csv(out_path, index=False)
    print(f"Saved {out_path.name}: {len(df_year)} rows\n")


# Load the boundary polygons for the communities

# In[ ]:


def load_boundary(path):
    # header=None + to_numeric handles files with or without a header row
    b = pd.read_csv(path, header=None, names=["lat", "lon"])
    b = b.apply(pd.to_numeric, errors="coerce").dropna()
    return list(zip(b["lat"], b["lon"]))


boundaries = {name: load_boundary(p) for name, p in boundary_files.items()}

for name, pts in boundaries.items():
    print(f"{name}: {len(pts)} vertices")
decoded_files = sorted(dec_dir.glob("*_decoded.csv"))

# Build the map

# In[ ]:


tiles_url = (
    "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_50ry_1_cccdef19d6d2d84e86a24f0b"
)

# -----------------------------------
# 2. Create base map
# -----------------------------------
df = pd.read_csv(in_dir / "regional_h3_coord_bandwidth.csv")
df.columns = df.columns.str.strip()

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


for provider in providers:
    for (mindown, minup), style in speed_tiers.items():
        subset = df[
            (df["providerid"] == provider)
            & (df["mindown"] == mindown)
            & (df["minup"] == minup)
        ].drop_duplicates(subset="h3_res9_id")

        if subset.empty:
            print(f"Provider {provider} {mindown}/{minup}: no data, skipped")
            continue

        color = provider_colors[provider][style["shade"]]
        title = f"Provider {provider} - {mindown}/{minup} Mbps"
        m = build_map(subset, color, style["opacity"], title)

        map_out_path = map_dir / f"map_{provider}_{mindown}_{minup}.html"
        m.save(map_out_path)
        print(f"Map created: {map_out_path.name} ({len(subset)} cells)")


# Build the historical maps

# In[ ]:


# Historical maps show Cuba only, so overlay just the Cuba boundary and town marker
cuba_boundaries = {"Cuba, NM": boundaries["Cuba, NM"]}
cuba_towns      = {"Cuba, NM": towns["Cuba, NM"]}

for year in (2025, 2022):
    df_year = pd.read_csv(hist_dir / f"cuba_h3_coord_bandwidth_{year}.csv")
    df_year.columns = df_year.columns.str.strip()
    df_year["h3_res9_id"] = df_year["h3_res9_id"].astype(str)

    for provider in providers:
        for (mindown, minup), style in speed_tiers.items():
            subset = df_year[
                (df_year["providerid"] == provider)
                & (df_year["mindown"] == mindown)
                & (df_year["minup"] == minup)
            ].drop_duplicates(subset="h3_res9_id")

            if subset.empty:
                print(f"{year} Provider {provider} {mindown}/{minup}: no data, skipped")
                continue

            color = provider_colors[provider][style["shade"]]
            title = f"Cuba {year} - Provider {provider} - {mindown}/{minup} Mbps"
            m = build_map(subset, color, style["opacity"], title,
                          boundaries=cuba_boundaries, towns=cuba_towns)

            map_out_path = map_dir / f"map_cuba_{year}_{provider}_{mindown}_{minup}.html"
            m.save(map_out_path)
            print(f"Map created: {map_out_path.name} ({len(subset)} cells)")


# Determine coverage and plot

# In[ ]:


def cells_in_boundary(pts, res=9):
    poly = h3.LatLngPoly(pts)          # (lat, lon) pairs, no need to close the ring
    return set(h3.h3shape_to_cells(poly, res))

town_cells = {name: cells_in_boundary(pts) for name, pts in boundaries.items()}

for name, cells in town_cells.items():
    print(f"{name}: {len(cells)} cells ({len(cells) * CELL_KM2:.1f} km2)")

# -----------------------------------
# 3. Load coverage data and index it once
# -----------------------------------
df = pd.read_csv(in_dir / "regional_h3_coord_bandwidth.csv")
df.columns = df.columns.str.strip()
df["h3_res9_id"] = df["h3_res9_id"].astype(str)

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
# 5. Save and display
# -----------------------------------
out_path = in_dir / "coverage_summary.csv"
summary.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}\n")

print(summary.pivot_table(
    index=["town", "provider"],
    columns="tier",
    values="pct_covered",
))

# -----------------------------------
# 6. Bar chart
# -----------------------------------
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

chart_path = in_dir / "coverage_bar_chart.png"
fig.savefig(chart_path, dpi=200)
plt.show()
print(f"Saved: {chart_path}")


# Line chart showing coverage trend over data from 2022, 2025, and 2026:

# In[ ]:


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
    "2022": build_lookup(pd.read_csv(hist_dir / "cuba_h3_coord_bandwidth_2022.csv")),
    "2025": build_lookup(pd.read_csv(hist_dir / "cuba_h3_coord_bandwidth_2025.csv")),
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
trend.to_csv(in_dir / "cuba_coverage_over_time.csv", index=False)
print(trend.pivot_table(index=["provider", "tier"], columns="period",
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

line_path = in_dir / "cuba_coverage_line_chart.png"
fig.savefig(line_path, dpi=200)
plt.show()
print(f"Saved: {line_path}")


# In[ ]:




