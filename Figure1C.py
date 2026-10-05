# Figure 1: Boundary/ interior intensity ratio (488 + 594)
# Metric: mean of top 10% brightest boundary ring pixels (bgsub) / interior mean (bgsub).
# requires boundary_pixels_488.csv, inside_pixels_488.csv, boundary_pixels_594.csv, inside_pixels_594.csv  
# /Users/aleksalakic/Desktop/Figure1C_FINAL/

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

root_path = open('experimental_data/experiment_index.txt', 'r').readlines()[0].strip()
input_folder = f'{root_path}Results_2/'
output_folder = f'{root_path}Figures/'
os.makedirs(output_folder, exist_ok=True)


metric = 'coating_ratio'
ylabel_text = 'Boundary / interior intensity'


def load_and_process(boundary_file, inside_file):
    bp = pd.read_csv(f"{input_folder}/{boundary_file}")
    ip = pd.read_csv(f"{input_folder}/{inside_file}")
    df = bp.merge(ip, on=['stack', 'frame', 'droplet_id'], suffixes=('_bp', '_ip'))

    parts = df['stack'].str.split('_', expand=True)
    df['method'] = parts[1]
    df['oil'] = parts[2].str.lower()
    df['protein'] = parts[3].str.lower()
    df['replicate'] = parts[4]

    df = df[['oil', 'protein', 'replicate', metric]].dropna()
    df = df.groupby(['oil', 'protein', 'replicate']).mean().reset_index()
    df['oil_order'] = df['oil']
    return df


bp_488 = load_and_process('boundary_pixels_488.csv', 'inside_pixels_488.csv')
bp_647 = load_and_process('boundary_pixels_647.csv', 'inside_pixels_647.csv')
bp_488['channel'] = '488'
bp_647['channel'] = '647'

print(f"488: {len(bp_488)} replicate-averaged points")
print(f"647: {len(bp_647)} replicate-averaged points")

panel_a = pd.concat([
    bp_488[bp_488['protein'] == 'dendra'].assign(x_label='Dendra (488)'),
    bp_647[bp_647['protein'] == 'mcherry'].assign(x_label='mCherry (647)')
], ignore_index=True)

panel_b = pd.concat([
    bp_488[bp_488['protein'] == 'dendra+mcherry'].assign(x_label='Dendra+mCherry (488)'),
    bp_647[bp_647['protein'] == 'dendra+mcherry'].assign(x_label='Dendra+mCherry (647)')
], ignore_index=True)


def bardotplot(data, xcol, ycol, order, hue, hue_order, palette, ax,
               xlabel='', ylabel='', dot_size=5, cap_size=0.15, cap_width=1):
    sns.barplot(data=data, x=xcol, y=ycol, hue=hue, palette=palette,
                capsize=cap_size, errwidth=cap_width, ax=ax, dodge=True,
                order=order, hue_order=hue_order, edgecolor='white')
    sns.stripplot(data=data, x=xcol, y=ycol, hue=hue, palette=palette, ax=ax,
                  edgecolor='#fff', linewidth=1, s=dot_size, order=order,
                  hue_order=hue_order, dodge=True)

    n_hue = len(hue_order)
    bar_width = 0.8 / n_hue

    for i, x_group in enumerate(order):
        subset = data[data[xcol] == x_group]
        grp_data = {name: grp[ycol].values for name, grp in subset.groupby(hue)
                    if len(grp[ycol].values) >= 2 and name in hue_order}
        if len(grp_data) < 2:
            continue

        f_stat, anova_p = stats.f_oneway(*grp_data.values())
        print(f"  {x_group}: ANOVA F={f_stat:.2f}, p={anova_p:.4e}")

        pair_keys = [(a, b) for a, b in combinations(hue_order, 2)
                     if a in grp_data and b in grp_data]
        raw_pvals = [stats.ttest_ind(grp_data[a], grp_data[b])[1] for a, b in pair_keys]
        if not raw_pvals:
            continue
        _, corrected_pvals, _, _ = multipletests(raw_pvals, method='holm')

        y_max = subset[ycol].max()
        y_ax_max = ax.get_ylim()[1]
        if y_ax_max and y_max > y_ax_max:
            y_max = y_ax_max * 0.75
        y_step = y_max * 0.10
        y_start = y_max * 1.10

        for idx, (a, b) in enumerate(pair_keys):
            p_corr = corrected_pvals[idx]
            if p_corr < 0.0001:
                p_text = '****'
            elif p_corr < 0.001:
                p_text = '***'
            elif p_corr < 0.01:
                p_text = '**'
            elif p_corr < 0.05:
                p_text = '*'
            else:
                p_text = 'ns'

            idx_a = hue_order.index(a)
            idx_b = hue_order.index(b)
            x_a = i - 0.8/2 + bar_width * (idx_a + 0.5)
            x_b = i - 0.8/2 + bar_width * (idx_b + 0.5)
            y_bracket = y_start + idx * y_step
            bracket_h = y_max * 0.02
            ax.plot([x_a, x_a, x_b, x_b],
                    [y_bracket - bracket_h, y_bracket, y_bracket, y_bracket - bracket_h],
                    color='black', linewidth=0.5, clip_on=False)
            ax.text((x_a + x_b) / 2, y_bracket + bracket_h * 0.3, p_text,
                    ha='center', va='bottom', fontsize=6, clip_on=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend('', frameon=False)
    return ax


plt.rc('font', family='arial', weight='normal', size=8)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm = 1/2.54

hue_order = ['oil', 'ol', 'peglipid']
panel_a_order = ['Dendra (488)', 'mCherry (647)']
panel_b_order = ['Dendra+mCherry (488)', 'Dendra+mCherry (647)']

data_max = max(panel_a[metric].max(), panel_b[metric].max())
y_upper = np.ceil(data_max * 1.3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18*cm, 7*cm), sharey=True)
ax1.set_ylim(0, 16)

print("\n=== Panel A ===")
bardotplot(panel_a, 'x_label', metric, panel_a_order, 'oil_order', hue_order,
           'Set2', ax1, ylabel=ylabel_text)
#ax1.axhline(1.0, color='black', linestyle=':', linewidth=0.7, alpha=0.6, zorder=0)
ax1.set_title('Single Protein', fontsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

print("\n=== Panel B ===")
bardotplot(panel_b, 'x_label', metric, panel_b_order, 'oil_order', hue_order,
           'Set2', ax2, ylabel='')
#ax2.axhline(1.0, color='black', linestyle=':', linewidth=0.7, alpha=0.6, zorder=0)
ax2.set_title('Co-encapsulated', fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(handles=by_label.values(), labels=by_label.keys(), title='Oil Type',
           loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=7, title_fontsize=7,
           framealpha=0.9, edgecolor='lightgray')

plt.tight_layout()

for label, data in [('Panel A', panel_a), ('Panel B', panel_b)]:
    print(f"\n{'='*50}\nTukey HSD: {label}\n{'='*50}")
    for x_group in data['x_label'].unique():
        subset = data[data['x_label'] == x_group]
        if subset['oil_order'].nunique() >= 2:
            tukey = pairwise_tukeyhsd(subset[metric], subset['oil_order'], alpha=0.05)
            print(f"\n{x_group}:\n{tukey}")

output_path = os.path.join(output_folder, f'boundary_{metric}.svg')
plt.savefig(output_path, bbox_inches='tight', dpi=300, format='svg')
print(f"\nSaved to {output_path}")