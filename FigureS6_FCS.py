# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import seaborn as sns

# ======================================================================
# STYLE
# ======================================================================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
})
sns.set_theme(style="ticks")

# ======================================================================
# PATH
# ======================================================================
FCS_DIR = r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\FCS"

# ======================================================================
# LOAD FILES
# ======================================================================
all_files = [
    os.path.join(FCS_DIR, f)
    for f in os.listdir(FCS_DIR)
    if f.endswith(".xlsx")
]

condition_groups = defaultdict(list)

for fp in all_files:
    name = os.path.basename(fp).lower()

    if 'dibbot' not in name:
        continue

    # exclude mcherry lysate
    if 'mcherry' in name and 'lysate' in name:
        continue

    if 'dendra' in name:
        condition_groups['dendra'].append(fp)
    elif 'mcherry' in name:
        condition_groups['mcherry'].append(fp)

colors = {
    'dendra': '#9B59B6',
    'mcherry': '#E7549E'
}

# ======================================================================
# PROCESS DATA
# ======================================================================
all_plot_data = []

for cond, files in condition_groups.items():
    dfs = [pd.read_excel(fp) for fp in files]

    all_times = np.unique(np.concatenate([df['Time [ms]'].values for df in dfs]))
    common_time = np.sort(all_times)

    for df in dfs:
        for rep in df['Replicate'].unique():
            df_rep = df[df['Replicate'] == rep]
            vals = df_rep.set_index('Time [ms]')['Value']

            if vals.empty:
                continue

            interp_vals = np.interp(common_time, vals.index, vals.values)

            max_val = np.nanmax(interp_vals)
            if max_val != 0:
                interp_vals = interp_vals / max_val

            all_plot_data.append(pd.DataFrame({
                'Time [ms]': common_time,
                'Value': interp_vals,
                'Condition': cond
            }))

df_fcs = pd.concat(all_plot_data, ignore_index=True)

# ======================================================================
# PLOT (10 cm x 6 cm)
# ======================================================================
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(18/2.54, 10/2.54)
)

# ---- Dendra ----
sns.lineplot(
    data=df_fcs[df_fcs['Condition'] == 'dendra'],
    x='Time [ms]',
    y='Value',
    errorbar=('sd', 1),
    color=colors['dendra'],
    ax=ax1
)

ax1.set_xscale('log')
ax1.set_ylim(0, 1.05)
ax1.set_title('GST-Dendra2')
ax1.set_xlabel('Time [ms]')
ax1.set_ylabel('Normalised G(t)')
ax1.spines[['top', 'right']].set_visible(False)

ax1.text(
    -0.15, 1.05, 'A',
    transform=ax1.transAxes,
    fontsize=12,
    fontweight='bold',
    va='top',
    ha='left'
)

# ---- mCherry ----
sns.lineplot(
    data=df_fcs[df_fcs['Condition'] == 'mcherry'],
    x='Time [ms]',
    y='Value',
    errorbar=('sd', 1),
    color=colors['mcherry'],
    ax=ax2
)

ax2.set_xscale('log')
ax2.set_ylim(0, 1.05)
ax2.set_title('mCherry')
ax2.set_xlabel('Time [ms]')
ax2.set_ylabel('Normalised G(t)')
ax2.spines[['top', 'right']].set_visible(False)

ax2.text(
    -0.15, 1.05, 'B',
    transform=ax2.transAxes,
    fontsize=12,
    fontweight='bold',
    va='top',
    ha='left'
)

plt.tight_layout()
plt.show()
# %%
svg_path = os.path.join(OUTPUT_DIR, "SuppFIGURE1.svg")
fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
print(f"Saved panel plot → {svg_path}")









# %%
