import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import math

plt.ion()

# ============================================================================
# FIGURE SETTINGS
# ============================================================================
sns.set_theme(style="white", context="notebook")
font = {'family': 'arial', 'weight': 'normal', 'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm_unit = 1 / 2.54

# === Define colors ===
colors = {
    "DIB-BOT": "#E75480",  # bright pink-magenta
    "Well":    "#00CC66",  # green
}

# ============================================================================
# PATHS
# ============================================================================
root_path = open('experimental_data/experiment_index.txt', 'r').readlines()[0].strip()
data_dir = f'{root_path}/'
output_folder = f'{root_path}Figures/'
os.makedirs(output_folder, exist_ok=True)

# ============================================================================
# DATA MANIFEST
# ============================================================================
# Each entry: (filename, sample_label, group)
# group: 'droplet' (dibbot) or 'well'
manifest = [
    ('aGFP_dibbot_combined.csv',            'aGFP',           'DIB-BOT'),
    ('aGFP_well_combined.csv',              'aGFP',           'Well'),
    ('Dendra_dibbot_combined.csv',          'Dendra',         'DIB-BOT'),
    ('Dendra_well_combined.csv',            'Dendra',         'Well'),
    ('mCherry_dibbot_combined.csv',         'mCherry',        'DIB-BOT'),
    ('mCherry_well_combined.csv',           'mCherry',        'Well'),
    ('mCherry_lysate_dibbot_combined.csv',  'mCherry lysate', 'DIB-BOT'),
    ('mCherry_lysate_well_combined.csv',    'mCherry lysate', 'Well'),
    ('SRB_dibbot_combined.csv',             'SRB',            'DIB-BOT'),
]

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================
def load_combined_csv(file_path: str) -> pd.DataFrame:
    """Load a pre-combined FCS CSV file."""
    df = pd.read_csv(file_path)
    required = {'logtime', 'value_norm'}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns {required - set(df.columns)} in {file_path}")
    return df


def plot_fcs_trace(ax, df, color, label):
    """Plot mean ± SD of normalised FCS curves on a given axis."""
    m = df.groupby('logtime', as_index=False)['value_norm'].mean()
    s = df.groupby('logtime', as_index=False)['value_norm'].std()

    ax.plot(m['logtime'], m['value_norm'], linewidth=2, color=color, label=label)
    ax.fill_between(m['logtime'],
                    m['value_norm'] - s['value_norm'],
                    m['value_norm'] + s['value_norm'],
                    color=color, alpha=0.15)

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"Looking for CSVs in: {data_dir}")
print(f"Files found: {os.listdir(data_dir) if os.path.isdir(data_dir) else 'DIRECTORY NOT FOUND'}")

# Store as {sample_label: {group: DataFrame}}
data = {}
for filename, label, group in manifest:
    fpath = os.path.join(data_dir, filename)
    if not os.path.isfile(fpath):
        print(f"⚠ Skipping (not found): {fpath}")
        continue
    df = load_combined_csv(fpath)
    data.setdefault(label, {})[group] = df

if not data:
    raise FileNotFoundError(
        f"No CSV files matched in {data_dir}. "
        "Check that data_dir points to the folder containing the _combined.csv files."
    )

# ============================================================================
# SPLIT: paired samples vs SRB (droplet only)
# ============================================================================
paired_samples = ['aGFP', 'Dendra', 'mCherry', 'mCherry lysate']
paired_samples = [s for s in paired_samples if s in data]

# ============================================================================
# FIGURE 1 — Droplet vs Well overlay (paired samples)
# ============================================================================
n = len(paired_samples)
ncols = 2
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
axes = np.array(axes).reshape(-1)

for i, sample in enumerate(paired_samples):
    ax = axes[i]
    for group in ['DIB-BOT', 'Well']:
        if group in data[sample]:
            plot_fcs_trace(ax, data[sample][group], color=colors[group], label=group)
    ax.set_title(sample)
    ax.set_xlabel("log(Time ms)")
    ax.set_ylabel("Normalized G(τ)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)

for ax in axes[n:]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()

svg_path = os.path.join(output_folder, 'fcs_droplet_vs_well.svg')
fig.savefig(svg_path, format='svg', bbox_inches='tight')
print(f"Figure saved: {svg_path}")

# ============================================================================
# FIGURE 2 — SRB (droplet only)
# ============================================================================
if 'SRB' in data:
    fig_srb, ax_srb = plt.subplots(1, 1, figsize=(4, 3))
    if 'DIB-BOT' in data['SRB']:
        plot_fcs_trace(ax_srb, data['SRB']['DIB-BOT'],
                       color=colors['DIB-BOT'], label='DIB-BOT')
    if 'Well' in data['SRB']:
        plot_fcs_trace(ax_srb, data['SRB']['Well'],
                       color=colors['Well'], label='Well')
    ax_srb.set_title('SRB')
    ax_srb.set_xlabel("log(Time ms)")
    ax_srb.set_ylabel("Normalized G(τ)")
    ax_srb.legend(fontsize=7)
    ax_srb.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()

    svg_path_srb = os.path.join(output_folder, 'fcs_SRB.svg')
    fig_srb.savefig(svg_path_srb, format='svg', bbox_inches='tight')
    print(f"Figure saved: {svg_path_srb}")