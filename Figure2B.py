# Joint plot: Droplet eccentricity vs physical area by method
# Uses scale-corrected area from droplet_char.csv
# Reports CV (area), mean ± SD (eccentricity), KS test, Mann-Whitney U
# /Users/aleksalakic/Desktop/Figure2_23-04-26/Figure2B_analysis

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu

# ============================================================================
# FIGURE SETTINGS
# ============================================================================
font = {'family': 'arial', 'weight': 'normal', 'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm = 1/2.54

# ============================================================================
# PATHS
# ============================================================================
root_path = open('experimental_data/experiment_index.txt', 'r').readlines()[0].strip()
input_folder = f'{root_path}/Results/'
output_folder = f'{root_path}/Figures/'

os.makedirs(output_folder, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
char_path = f"{input_folder}/droplet_char.csv"
char_df = pd.read_csv(char_path)

print("Columns available:", char_df.columns.tolist())
print(f"Total rows: {len(char_df)}")

def extract_method(stack_name):
    s = stack_name.lower()
    fields = s.split('_')
    if 'hp' in fields:
        return 'HP'
    elif 'dibbot' in fields:
        return 'dibbot'
    return 'unknown'

char_df['method'] = char_df['stack'].apply(extract_method)
char_df['date'] = char_df['stack'].str.split('_').str[1]

print("\nUnique methods:", char_df['method'].unique())

# ============================================================================
# CHECK SCALE AVAILABILITY
# ============================================================================
has_physical = 'area_um2' in char_df.columns and char_df['area_um2'].notna().any()

if has_physical:
    area_col = 'area_um2'
    area_label = 'Droplet Area (µm²)'
    area_unit = 'µm²'
    print(f"\nUsing physical area (µm²)")
    print(f"  Rows with scale: {char_df['area_um2'].notna().sum()}")
    print(f"  Rows without scale: {char_df['area_um2'].isna().sum()}")
else:
    area_col = 'area'
    area_label = 'Droplet Area (px)'
    area_unit = 'px'
    print("\nWARNING: area_um2 not available — falling back to pixel area")

# ============================================================================
# FILTER DATA
# ============================================================================
methods = ['dibbot', 'hp']
df_plot = char_df[char_df['method'].str.lower().isin(methods)].copy()
df_plot = df_plot.loc[df_plot.groupby(['stack', 'frame'])['area'].idxmax()]

if has_physical:
    before = len(df_plot)
    df_plot = df_plot.dropna(subset=['area_um2'])
    after = len(df_plot)
    if before != after:
        print(f"Dropped {before - after} rows with missing scale")

print(f"\nFiltered to {len(df_plot)} droplets")

label_map = {}
for m in df_plot['method'].unique():
    if m.lower() == 'dibbot':
        label_map[m] = 'DIB-BOT'
    elif m.lower() == 'hp':
        label_map[m] = 'Hand Pipette'
    else:
        label_map[m] = m
df_plot['Method'] = df_plot['method'].map(label_map)

# ============================================================================
# DESCRIPTIVE STATS: CV (area) and mean ± SD (eccentricity)
# CV reported for area (physically interpretable: dispensing reproducibility).
# Eccentricity is bounded [0,1] and DIB-BOT values are near zero, where CV is
# inflated by a small denominator and not meaningful — report mean ± SD instead.
# ============================================================================
stats_summary = {}
for method in sorted(df_plot['Method'].unique()):
    subset = df_plot[df_plot['Method'] == method]
    area_vals = subset[area_col].values
    ecc_vals = subset['eccentricity'].values

    area_mean = area_vals.mean()
    area_sd = area_vals.std(ddof=1)
    area_cv = 100 * area_sd / area_mean
    ecc_mean = ecc_vals.mean()
    ecc_sd = ecc_vals.std(ddof=1)
    ecc_median = np.median(ecc_vals)
    ecc_iqr = np.percentile(ecc_vals, 75) - np.percentile(ecc_vals, 25)

    stats_summary[method] = {
        'n': len(subset),
        'area_mean': area_mean,
        'area_sd': area_sd,
        'area_cv': area_cv,
        'ecc_mean': ecc_mean,
        'ecc_sd': ecc_sd,
        'ecc_median': ecc_median,
        'ecc_iqr': ecc_iqr,
    }

    print(f"\n{method} (n = {len(subset)}):")
    print(f"  Area:         {area_mean:.2f} ± {area_sd:.2f} {area_unit}   CV = {area_cv:.1f}%")
    print(f"  Eccentricity: {ecc_mean:.3f} ± {ecc_sd:.3f}   "
          f"median = {ecc_median:.3f} (IQR {ecc_iqr:.3f})")

# ============================================================================
# STATISTICAL TESTS
# KS: distribution-level difference in eccentricity
# Mann-Whitney U: location shift in eccentricity (non-parametric, bounded data)
# ============================================================================
groups = sorted(df_plot['Method'].unique())
ks_annot = None
if len(groups) == 2:
    a_ecc = df_plot[df_plot['Method'] == groups[0]]['eccentricity'].values
    b_ecc = df_plot[df_plot['Method'] == groups[1]]['eccentricity'].values

    d_stat, p_ks = ks_2samp(a_ecc, b_ecc)
    u_stat, p_mwu = mannwhitneyu(a_ecc, b_ecc, alternative='two-sided')

    print(f"\nEccentricity comparison ({groups[0]} vs {groups[1]}):")
    print(f"  KS:           D = {d_stat:.4f}, p = {p_ks:.4g}")
    print(f"  Mann-Whitney: U = {u_stat:.1f}, p = {p_mwu:.4g}")

    p_str = f"p = {p_ks:.1e}" if p_ks < 1e-3 else f"p = {p_ks:.3f}"
    ks_annot = f"KS (eccentricity): D = {d_stat:.3f}, {p_str}"

# ============================================================================
# WRITE SUMMARY TO TXT FOR MANUSCRIPT
# ============================================================================
summary_path = f"{output_folder}/droplet_stats_summary.txt"
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("Droplet area and eccentricity statistics by method\n")
    f.write("=" * 60 + "\n\n")
    for method, s in stats_summary.items():
        f.write(f"{method} (n = {s['n']})\n")
        f.write(f"  Area:         {s['area_mean']:.2f} ± {s['area_sd']:.2f} {area_unit}   "
                f"CV = {s['area_cv']:.1f}%\n")
        f.write(f"  Eccentricity: {s['ecc_mean']:.3f} ± {s['ecc_sd']:.3f}   "
                f"median = {s['ecc_median']:.3f} (IQR {s['ecc_iqr']:.3f})\n\n")
    if len(groups) == 2:
        f.write(f"Eccentricity comparison ({groups[0]} vs {groups[1]}):\n")
        f.write(f"  KS:           D = {d_stat:.4f}, p = {p_ks:.4g}\n")
        f.write(f"  Mann-Whitney: U = {u_stat:.1f}, p = {p_mwu:.4g}\n")

print(f"\nStats summary written to {summary_path}")

# ============================================================================
# JOINT PLOT
# ============================================================================
colors = {'DIB-BOT': "#318989", 'Hand Pipette': "#cf991a"}

g = sns.JointGrid(
    data=df_plot,
    x=area_col,
    y='eccentricity',
    hue='Method',
    palette=colors,
    height=7*cm,
    ratio=4,
    marginal_ticks=False,
)

g.plot_joint(
    sns.scatterplot,
    s=50,
    alpha=0.65,
    edgecolor='white',
    linewidth=0.7,
    legend='full',
)

g.plot_marginals(
    sns.kdeplot,
    fill=True,
    alpha=0.3,
    linewidth=1,
    common_norm=False,
    clip=(0, None),
)

g.ax_joint.set_xlabel(area_label)
g.ax_joint.set_ylabel('Eccentricity')
g.ax_joint.set_ylim(-0.02, None)
g.ax_joint.spines['top'].set_visible(False)
g.ax_joint.spines['right'].set_visible(False)

for ax in [g.ax_marg_x, g.ax_marg_y]:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

if ks_annot is not None:
    g.ax_joint.text(
        0.98, 0.98, ks_annot,
        transform=g.ax_joint.transAxes,
        ha='right', va='top',
        fontsize=7,
    )

legend = g.ax_joint.get_legend()
if legend:
    legend.set_frame_on(False)
    for text in legend.get_texts():
        text.set_fontsize(7)
    legend.set_title('')

g.fig.subplots_adjust(top=0.92)
g.fig.suptitle('Droplet Area vs Eccentricity', fontsize=8, y=0.97)

g.savefig(f'{output_folder}/jointplot_eccentricity_area.svg')
g.savefig(f'{output_folder}/jointplot_eccentricity_area.png', dpi=300)
plt.show()

print(f"\nFigure saved to {output_folder}")