# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

# ====================================================================
# STYLE
# ======================================================================
font = {'family': 'arial', 'weight': 'normal', 'size': 10}
matplotlib.rc('font', **font)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300

cm = 1 / 2.54
sns.set_theme(style="ticks")

# ======================================================================
# PATH
# ======================================================================
DATA_DIR = r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\Supplementary paper\FL1_0_and_30_mins"

files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".xlsx")
]

# ======================================================================
# GROUP FILES (EXACT MATCH)
# ======================================================================
groups = {
    "0 min": [],
    "30 min": []
}

for fp in files:
    name = os.path.basename(fp)

    if name == "260311_FL1_0min_cleaned.xlsx":
        groups["0 min"].append(fp)

    elif name == "260311_FL1_30min_cleaned.xlsx":
        groups["30 min"].append(fp)

# ======================================================================
# CHANNEL COLOURS
# ======================================================================
channel_colors = {
    'Fit Channel 1': '#9B59B6',
    'Fit Channel 2': '#E7549E',
    'Fit Channel 1 -> 2': '#00A6D6'
}

# ======================================================================
# SAFE FILE READER (FIXES YOUR TIME ERROR)
# ======================================================================
def read_clean_file(fp):
    df = pd.read_excel(fp)
    df.columns = df.columns.str.strip()

    # find time column safely
    time_col = None
    for c in df.columns:
        if "time" in c.lower():
            time_col = c
            break

    if time_col is None:
        raise ValueError(f"No time column in {fp}")

    df = df.rename(columns={time_col: "Time [ms]"})
    return df

# ======================================================================
# PROCESS (same logic as Panel B)
# ======================================================================
def process_files(file_list):

    if len(file_list) == 0:
        return pd.DataFrame(columns=['Time [ms]', 'Value', 'Channel'])

    dfs = [read_clean_file(fp) for fp in file_list]

    all_times = np.unique(np.concatenate([df['Time [ms]'].values for df in dfs]))
    common_time = np.sort(all_times)

    aligned_dfs = []

    for df in dfs:
        replicates = df['Replicate'].unique() if 'Replicate' in df.columns else [0]

        for rep in replicates:
            df_rep = df[df['Replicate'] == rep] if 'Replicate' in df.columns else df

            df_aligned = pd.DataFrame({'Time [ms]': common_time})

            for ch in df['Channel'].unique():
                vals = df_rep[df_rep['Channel'] == ch].set_index('Time [ms]')['Value']

                if vals.empty:
                    continue

                df_aligned[ch] = np.interp(common_time, vals.index, vals.values)

            df_long = df_aligned.melt(
                id_vars='Time [ms]',
                value_vars=[c for c in df['Channel'].unique() if c in df_aligned.columns],
                var_name='Channel',
                value_name='Value'
            )

            aligned_dfs.append(df_long)

    return pd.concat(aligned_dfs, ignore_index=True)

# ======================================================================
# FIGURE (18 cm × 10 cm)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(18 * cm, 10 * cm))

titles = ["acGFP1-mCherry and Tev 0 min", "acGFP1-mCherry and Tev 30 min"]
keys = ["0 min", "30 min"]

for ax, title, key in zip(axes, titles, keys):

    df = process_files(groups[key])

    sns.lineplot(
        data=df,
        x='Time [ms]',
        y='Value',
        hue='Channel',
        errorbar=('sd', 1),
        palette=channel_colors,
        ax=ax
    )

    ax.set_xscale('log')
    ax.set_title(title)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("G(t)")

    ax.set_ylim(0, 0.0045)
    ax.set_yticks(np.linspace(0, 0.0045, 6))

    ax.set_xticks([1e-2, 1e0, 1e2], labels=['$10^{-2}$', '$10^{0}$', '$10^{2}$'])
    ax.xaxis.set_minor_locator(NullLocator())

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

# legend only on left
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, frameon=False, loc='upper right', fontsize=8)
axes[1].legend().remove()

# panel labels
axes[0].text(-0.15, 1.05, "A", transform=axes[0].transAxes,
             fontsize=10, fontweight='bold')

axes[1].text(-0.15, 1.05, "B", transform=axes[1].transAxes,
             fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()






# %%
svg_path = os.path.join(OUTPUT_DIR, "260522_Supp figure 2.svg")
fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
print(f"Saved panel plot → {svg_path}")
# %%
