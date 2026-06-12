import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ============================================================================
# FIGURE SETTINGS
# ============================================================================
sns.set_theme(style="white", context="notebook")
font = {'family': 'arial', 'weight': 'normal', 'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm = 1 / 2.54

# ============================================================================
# PATHS
# ============================================================================
root_path = open('experimental_data/experiment_index.txt', 'r').readlines()[0].strip()
output_folder = f'{root_path}Figures/'
os.makedirs(output_folder, exist_ok=True)

# --- LIST YOUR DATA FILES HERE (relative to root_path) --------------------
# Each entry: (relative_path, optional_sample_label)
# sample_label only needed if filename doesn't contain a recognisable tag
manifest = [
    # 250512
    ('250512_dibbot_Dendra+mCherry_curve_tidy.xlsx', None),
    ('250512_dibbot_FL1_curve_tidy.xlsx',            None),
    ('250512_dibbot_UT_curve_tidy.xlsx',             None),
    # 250514
    ('250514_dibbot_dendra+mCherry_droplet_curve_tidy.xlsx',      None),
    ('250514_dibbot_FL1_CellLysate_droplet_curve_tidy.xlsx',      None),
    ('250514_well_dendra+mCherry_curve_tidy.xlsx',                None),
    ('250514_well_FL1_CellLysate_droplet_curve_tidy.xlsx',        None),
    ('250514_well_UT_curve_tidy.xlsx',                            None),
    ('250514_dibbot_-veControl_PeglipidOil_tidy.xlsx',            'UT/Control'),
    # 250527
    ('250527_dibbot_dendra+mcherry_curve_tidy.xlsx', None),
    ('250527_dibbot_FL1_curve_tidy.xlsx',            None),
    ('250527_dibbot_ut_curve_tidy.xlsx',             None),
    ('250527_well_dendra+mcherry_curve_tidy.xlsx',   None),
    ('250527_well_FL1_curve_tidy.xlsx',              None),
    ('250527_well_ut_curve_tidy.xlsx',               None),
]

# ============================================================================
# LOAD & AGGREGATE DATA
# ============================================================================
all_long_list = []
for rel_path, lbl in manifest:
    fp = os.path.join(root_path, rel_path)
    df = pd.read_excel(fp)
    if lbl is not None:
        df['Sample'] = lbl
    if not df.empty:
        all_long_list.append(df)

all_long = pd.concat(all_long_list, ignore_index=True)

# Per-file means (biological replicate = one file)
per_file = (all_long
            .groupby(['File', 'Sample', 'Location', 'Channel', 'Time [ms]'], as_index=False)
            .agg(file_mean=('Value', 'mean')))

# Bio-rep mean ± SD across files
avg_biorep = (per_file
              .groupby(['Sample', 'Location', 'Channel', 'Time [ms]'], as_index=False)
              .agg(mean=('file_mean', 'mean'),
                   sd=('file_mean', 'std'),
                   n_bioreps=('file_mean', 'count')))

# ============================================================================
# CLEAN LABELS
# ============================================================================
# Rename channels for display
channel_labels = {
    'Fit/Green channel 1': 'Green autocorrelation',
    'Fit/Red channel 2':   'Red autocorrelation',
    'Fit channel 1>2':     'Cross-correlation',
}
avg_biorep['Channel'] = avg_biorep['Channel'].map(channel_labels).fillna(avg_biorep['Channel'])

# Rename samples for display
sample_labels = {
    'Dendra+mCherry': 'Dendra + mCherry',
    'FL1':            'FL1 (cell lysate)',
    'UT/Control':     'Untransfected',
}
avg_biorep['Sample'] = avg_biorep['Sample'].map(sample_labels).fillna(avg_biorep['Sample'])

# Define orders for the grid
sample_order   = ['Dendra + mCherry', 'FL1 (cell lysate)', 'Untransfected']
location_order = ['Dibbot', 'Well']
channel_order  = ['Green autocorrelation', 'Red autocorrelation', 'Cross-correlation']

# Colour palette
pal = {
    'Green autocorrelation': '#9B59B6',  # Purple
    'Red autocorrelation':   '#E7549E',  # Pink
    'Cross-correlation':     '#00A6D6',  # Blue
}

# ============================================================================
# PLOTTING — rows = Location, cols = Sample
# ============================================================================
# Filter to only conditions present in data
locations_present = [l for l in location_order if l in avg_biorep['Location'].unique()]
samples_present   = [s for s in sample_order   if s in avg_biorep['Sample'].unique()]
channels_present  = [c for c in channel_order  if c in avg_biorep['Channel'].unique()]

nrows = len(locations_present)
ncols = len(samples_present)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(ncols * 5 * cm + 4, nrows * 4.5 * cm + 2),
    sharex=True,
    sharey=False,
    squeeze=False,
)

for i, loc in enumerate(locations_present):
    for j, samp in enumerate(samples_present):
        ax = axes[i, j]
        sub = avg_biorep[(avg_biorep['Location'] == loc) &
                         (avg_biorep['Sample'] == samp)]

        for ch in channels_present:
            ch_data = sub[sub['Channel'] == ch].sort_values('Time [ms]')
            if ch_data.empty:
                continue
            t = ch_data['Time [ms]'].values
            m = ch_data['mean'].values
            s = ch_data['sd'].values

            ax.plot(t, m, color=pal[ch], linewidth=1)
            if np.any(np.isfinite(s)):
                ax.fill_between(t, m - s, m + s, alpha=0.2, color=pal[ch])

        ax.set_xscale('log')

        # Column titles (top row only)
        if i == 0:
            ax.set_title(samp, fontsize=8, fontweight='bold')

        # Row labels (left column only)
        if j == 0:
            ax.set_ylabel(f'{loc}\nG(\u03c4)', fontsize=7)
        else:
            ax.set_ylabel('')

        # X-axis label (bottom row only)
        if i == nrows - 1:
            ax.set_xlabel('\u03c4 (ms)', fontsize=7)
        else:
            ax.set_xlabel('')

        sns.despine(ax=ax)

# Shared legend along the bottom
handles = [Line2D([0], [0], color=pal[c], lw=2, label=c) for c in channels_present]
fig.legend(handles=handles, loc='lower center', ncol=len(channels_present),
           fontsize=7, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('FCCS correlation curves (bio-rep mean \u00b1 SD)',
             fontsize=9, fontweight='bold')
fig.tight_layout(rect=[0, 0.05, 1, 0.95])

fig.savefig(os.path.join(output_folder, 'fccs_combined_plot.svg'),
            format='svg', dpi=300, bbox_inches='tight')
plt.show()