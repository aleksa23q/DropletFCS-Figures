# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

# ============================================================================
# STYLE
# ============================================================================
font_settings = {
    'font.family': 'Arial',
    'font.weight': 'normal',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8
}
plt.rcParams.update(font_settings)
matplotlib.rcParams.update(font_settings)
sns.set_theme(style="ticks", context="notebook", rc=font_settings)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm = 1/2.54

# ============================================================================
# CONFIG
# ============================================================================
COUNT_REPLICATE_INDEX = 0  # 0 = R1, 1 = R2, ...

FOLDER = r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\Supplementary paper\For supp map_count"

# Condition -> filename, in the order you want them plotted/legended
FILES = {
    'PBS':        'mcherry PBS.xlsx',
    'UT':         'mcherry UT.xlsx',
    'well':       'mcherry well_260317.xlsx',
    'dibbot':     'mcherry dibbot_260319.xlsx',
    'coverslip':  'mcherry coverslip_260907.xlsx',
}

condition_order = ['PBS', 'UT', 'well', 'dibbot', 'coverslip']

condition_labels = {
    'PBS': 'PBS',
    'UT': 'Untransfected',
    'well': 'Chamber',
    'dibbot': 'DIB-BOT',
    'coverslip': 'Coverslip',
}

palette = {
    'PBS':       "#332F31",
    'UT':        "#AA5C5C",
    'well':      '#3d9973',
    'dibbot':    '#0496c7',
    'coverslip': '#C97064',
}

OUTPUT_DIR = r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\SVG_for_final_figures"

# ============================================================================
# COUNT-FILE READER
# ============================================================================
def read_count_replicate(fp, replicate_index=0):
    """Read one replicate's (Time, Count Rate) pair out of a '_count.xlsx'
    file. Each replicate occupies 2 columns (Time, Count Rate [kCounts/s]),
    with data starting on row index 2 (0-indexed) after two header rows."""
    if not os.path.isfile(fp):
        print(f"WARNING: file not found: {fp}")
        return None
    df_raw = pd.read_excel(fp, header=None)
    col_start = replicate_index * 2
    if col_start + 1 >= df_raw.shape[1]:
        print(f"WARNING: replicate index {replicate_index} out of range for {fp}")
        return None

    sub = df_raw.iloc[2:, [col_start, col_start + 1]].copy()
    sub.columns = ['Time', 'CountRate']
    sub = sub.dropna()
    sub = sub[pd.to_numeric(sub['Time'], errors='coerce').notna()]
    sub = sub.astype({'Time': float, 'CountRate': float})
    return sub

# ============================================================================
# LOAD
# ============================================================================
data = {}
for cond, fname in FILES.items():
    fp = os.path.join(FOLDER, fname)
    data[cond] = read_count_replicate(fp, COUNT_REPLICATE_INDEX)

# ============================================================================
# PLOT — Time vs Count Rate scatter, all 5 mCherry conditions on one panel
# ============================================================================
fig_width  = 12 * cm
fig_height = 8 * cm

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

for cond in condition_order:
    df = data.get(cond)
    if df is None or df.empty:
        continue
    ax.scatter(
        df['Time'], df['CountRate'],
        s=4, alpha=0.4,
        color=palette[cond],
        edgecolors='none',
        label=condition_labels[cond],
    )

ax.set_xlabel('Time [s]')
ax.set_ylabel('Count Rate [kCounts/s]')
ax.set_xlim(0, 10)
ax.set_title('mCherry')
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='both', which='both', direction='out', length=3, width=0.8)

legend = ax.legend(frameon=False, fontsize=8, markerscale=3, loc='upper right')

plt.tight_layout()
plt.show()
# %%
out_path = os.path.join(OUTPUT_DIR, "Figure_mcherry_count_rate_scatter_Supp1.svg")
fig.savefig(out_path, format="svg", dpi=300)
print(f"Saved to: {out_path}")