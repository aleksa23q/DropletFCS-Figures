"""
FCCS Plotter – TEV Protease Time Course
Generates 3 plots:
  1. Panel plot: All channels (autocorrelation + cross-correlation) per time point
  2. Overlay plot: Cross-correlation (Ch1→2) across time points with blue gradient
  3. Endpoint scatter: G(τ) at end of each cross-correlation curve vs time

All customization is in the SETTINGS section below.
"""

import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatter

# ============================================================================
# FIGURE SETTINGS
# ============================================================================
sns.set_theme(style="white", context="notebook")
font = {'family': 'arial', 'weight': 'normal', 'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm_unit = 1 / 2.54

# ============================================================================
# PATHS
# ============================================================================
root_path = open('experimental_data/experiment_index.txt', 'r').readlines()[0].strip()
output_folder = f'{root_path}FFigures/'
os.makedirs(output_folder, exist_ok=True)

# --- LIST YOUR DATA FILES HERE (relative to root_path) --------------------
# Each entry: (relative_path, time_in_minutes)
# time_in_minutes is used for ordering and labelling
manifest = [
    ('250702 dibbot 1in75 5 mins_cleaned.xlsx',  5),
    ('250702 dibbot 1in75 10 mins_cleaned.xlsx', 10),
    ('250702 dibbot 1in75 15 mins_cleaned.xlsx', 15),
    ('250702 dibbot 1in75 20 mins_cleaned.xlsx', 20),
    ('250702 dibbot 1in75 25 mins_cleaned.xlsx', 25),
    ('250702 dibbot 1in75 30 mins_cleaned.xlsx', 30),
    ('250702 dibbot 1in75 35 mins_cleaned.xlsx', 35),
]

# ============================================================================
# PLOT SETTINGS
# ============================================================================

# Channel colours
channel_colors = {
    'Fit Channel 1':      '#9B59B6',  # Purple  (autocorrelation ch1)
    'Fit Channel 2':      '#E7549E',  # Pink    (autocorrelation ch2)
    'Fit Channel 1 -> 2': '#00A6D6',  # Blue    (cross-correlation)
}

# Channel display labels
channel_labels = {
    'Fit Channel 1':      'Green autocorrelation',
    'Fit Channel 2':      'Red autocorrelation',
    'Fit Channel 1 -> 2': 'Cross-correlation',
}

# Panel plot
PANEL_FIG_WIDTH_PER_COL = 5   # cm
PANEL_FIG_HEIGHT_PER_ROW = 4.5  # cm

# Overlay plot (cross-correlation across timepoints)
OVERLAY_FIG_SIZE = (8, 5)
OVERLAY_CHANNEL = 'Fit Channel 1 -> 2'
OVERLAY_CMAP = 'Blues_r'
OVERLAY_LINE_WIDTH = 2

# Endpoint scatter
SCATTER_FIG_SIZE = (7.5, 5)
SCATTER_CMAP = 'Blues'
SCATTER_MARKER_SIZE = 80
SCATTER_XLIM = (0, 40)

# How to extract the scatter value from each cross-correlation curve:
#   'max'   = peak amplitude of the curve (highest G(τ))
#   'first' = G(τ) at shortest lag time
#   'last'  = G(τ) at longest lag time
ENDPOINT_METHOD = 'max'

# Spine / tick formatting (shared)
SPINE_WIDTH = 1.5
TICK_DIRECTION = 'out'
TICK_LENGTH = 6
TICK_WIDTH = 1.2

# ============================================================================
# LOAD DATA
# ============================================================================
file_times = []
for rel_path, t_min in manifest:
    fp = os.path.join(root_path, rel_path)
    file_times.append((t_min, fp))
file_times.sort(key=lambda x: x[0])

times = [t for t, _ in file_times]
print(f"Found {len(file_times)} files: {times} min")

# ============================================================================
# PLOT 1 – Panel plot (all channels per timepoint, excluding 35 min)
# ============================================================================
PANEL_NCOLS = 3
panel_times = [(t, fp) for t, fp in file_times if t != 35]
n = len(panel_times)
nrows = math.ceil(n / PANEL_NCOLS)

fig1, axes = plt.subplots(
    nrows, PANEL_NCOLS,
    figsize=(PANEL_NCOLS * PANEL_FIG_WIDTH_PER_COL * cm_unit + 4,
             nrows * PANEL_FIG_HEIGHT_PER_ROW * cm_unit + 2),
    sharex=True, sharey=True, squeeze=False,
)
axes_flat = axes.flatten()

for idx, (t, fp) in enumerate(panel_times):
    ax = axes_flat[idx]
    df = pd.read_excel(fp)

    for ch, color in channel_colors.items():
        df_ch = df[df['Channel'] == ch]
        if df_ch.empty:
            continue
        df_pivot = df_ch.pivot(index='Time [ms]', columns='Replicate', values='Value')
        mean_vals = df_pivot.mean(axis=1)
        std_vals = df_pivot.std(axis=1)
        ax.plot(df_pivot.index, mean_vals, color=color, linewidth=1)
        if np.any(np.isfinite(std_vals)):
            ax.fill_between(df_pivot.index, mean_vals - std_vals, mean_vals + std_vals,
                            alpha=0.2, color=color)

    ax.set_xscale('log')
    ax.set_title(f'{t} min', fontsize=8, fontweight='bold')

    # Y-axis label only on left column
    if idx % PANEL_NCOLS == 0:
        ax.set_ylabel('G(τ)', fontsize=7)
    else:
        ax.set_ylabel('')

    # X-axis label only on bottom row
    if idx >= n - PANEL_NCOLS:
        ax.set_xlabel('τ (ms)', fontsize=7)
    else:
        ax.set_xlabel('')

    sns.despine(ax=ax)

# Remove empty axes
for ax in axes_flat[n:]:
    fig1.delaxes(ax)

# Shared legend
handles = [Line2D([0], [0], color=v, lw=2, label=channel_labels.get(k, k))
           for k, v in channel_colors.items()]
fig1.legend(handles=handles, loc='lower center', ncol=len(channel_colors),
            fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))

fig1.suptitle('FCCS correlation curves (mean ± SD)', fontsize=9, fontweight='bold')
fig1.tight_layout(rect=[0, 0.05, 1, 0.95])

out1 = os.path.join(output_folder, 'panel_plot.svg')
fig1.savefig(out1, format='svg', dpi=300, bbox_inches='tight')
print(f"Saved → {out1}")

# ============================================================================
# PLOT 2 – Overlay of cross-correlation curves (time gradient)
# ============================================================================
cmap_overlay = plt.get_cmap(OVERLAY_CMAP)
vmin, vmax = min(times), max(times)
norm_cap = (25 - vmin) / (vmax - vmin) if vmax != vmin else 1.0

fig2, ax2 = plt.subplots(figsize=OVERLAY_FIG_SIZE)

for t, fp in file_times:
    df = pd.read_excel(fp)
    df_ch = df[df['Channel'].str.startswith(OVERLAY_CHANNEL)]
    if df_ch.empty:
        continue
    df_pivot = df_ch.pivot(index='Time [ms]', columns='Replicate', values='Value')
    mean_vals = df_pivot.mean(axis=1)
    std_vals = df_pivot.std(axis=1)

    frac = ((t - vmin) / (vmax - vmin)) * norm_cap if vmax != vmin else 0
    color = cmap_overlay(frac)

    ax2.plot(df_pivot.index, mean_vals, color=color, lw=OVERLAY_LINE_WIDTH,
             solid_capstyle='round', label=f"{t} min")
    ax2.fill_between(df_pivot.index, mean_vals - std_vals, mean_vals + std_vals,
                     color=color, alpha=0.3)

ax2.set_xscale('log')
ax2.set_xlabel('τ (ms)')
ax2.set_ylabel('G(τ)')
ax2.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
ax2.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
ax2.xaxis.set_major_formatter(LogFormatter())
ax2.legend(title="Time", fontsize=8, frameon=False)
sns.despine(ax=ax2)

fig2.tight_layout()
out2 = os.path.join(output_folder, 'overlay_cross_correlation.svg')
fig2.savefig(out2, format='svg', dpi=300, bbox_inches='tight')
print(f"Saved → {out2}")

# ============================================================================
# PLOT 3 – Endpoint scatter (G(τ) amplitude per cross-correlation curve)
# ============================================================================
endpoints = {}
for t, fp in file_times:
    df = pd.read_excel(fp)
    df_ch = df[df['Channel'].str.startswith(OVERLAY_CHANNEL)]
    if df_ch.empty:
        continue
    df_pivot = df_ch.pivot(index='Time [ms]', columns='Replicate', values='Value')
    mean_vals = df_pivot.mean(axis=1)
    if ENDPOINT_METHOD == 'max':
        endpoints[t] = mean_vals.max()
    elif ENDPOINT_METHOD == 'first':
        endpoints[t] = mean_vals.iloc[0]
    else:
        endpoints[t] = mean_vals.iloc[-1]

scatter_times = sorted(endpoints.keys())
scatter_vals = [endpoints[t] for t in scatter_times]

# Dark → light blue gradient (early = dark, late = light)
scatter_cmap = plt.get_cmap(SCATTER_CMAP)
scatter_norm = Normalize(vmin=min(scatter_times), vmax=max(scatter_times))
scatter_colors = [scatter_cmap(0.9 - scatter_norm(m) * 0.7) for m in scatter_times]

fig3, ax3 = plt.subplots(figsize=SCATTER_FIG_SIZE)
ax3.scatter(scatter_times, scatter_vals, c=scatter_colors, s=SCATTER_MARKER_SIZE,
            zorder=3)

ax3.set_xlabel('Time (min)')
ax3.set_ylabel('G(τ) at end of curve')
ax3.set_xticks(scatter_times)
ax3.set_xlim(*SCATTER_XLIM)
sns.despine(ax=ax3)

fig3.tight_layout()
out3 = os.path.join(output_folder, 'endpoint_scatter.svg')
fig3.savefig(out3, format='svg', dpi=300, bbox_inches='tight')
print(f"Saved → {out3}")

plt.show()
print(f"\nDone! All plots saved to: {output_folder}")