# Figure 1: peak boundary coating ratio across PEG-lipid conditions (488 nm)
# Metric: (boundary_mean_intensity - background) / inside_mean_intensity_bgsub
# >1 = boundary brighter than interior; ~1 = no coating.
#
# Reads  {root}Results/boundary_pixels_488.csv + inside_pixels_488.csv
# Writes {root}Figures/boundary_peak_boundary_ratio_488.svg (+ .png)
#        {root}Results/boundary_peak_boundary_ratio_488_stats.csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from itertools import combinations
from packaging.version import Version
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

root_path = open('experimental_data/experiment_index.txt', 'r').readlines()[0].strip()
input_folder = f'{root_path}Results/'
output_folder = f'{root_path}Figures/'
os.makedirs(output_folder, exist_ok=True)

channel = '488'
boundary_file = f'boundary_pixels_{channel}.csv'
inside_file = f'inside_pixels_{channel}.csv'

metric = 'peak_boundary_ratio'
ylabel_text = 'Peak boundary / interior intensity'

# Column names differ between processing versions; set them here rather than guessing.
# 'background' appears in both files, so the merge suffixes it to background_bp.
boundary_col = 'boundary_mean_intensity'
background_col = 'background_bp'
inside_col = 'inside_mean_intensity_bgsub'

# seaborn changed the categorical API in 0.12 (err_kws/legend vs errwidth/size)
sns_v12 = Version(sns.__version__) >= Version('0.12')

condition_order = ['oil', 'DPhPC', 'DPhPC_01PEG', 'DPhPC_05PEG',
                   'DPhPC_1PEG', 'DPhPC_5PEG', 'DPhPC_10PEG']

# 01PEG / 05PEG are 0.1% / 0.5% (zero-padded decimals)
condition_labels = {'oil': 'oil only', 'DPhPC': '0% PEG',
                    'DPhPC_01PEG': '0.1% PEG', 'DPhPC_05PEG': '0.5% PEG',
                    'DPhPC_1PEG': '1% PEG', 'DPhPC_5PEG': '5% PEG',
                    'DPhPC_10PEG': '10% PEG'}

# Brackets are drawn against this condition only; all pairs are still tested and written out.
reference_condition = 'DPhPC'

# 'droplet' pools I01/I02 of the same D-number, 'image' treats each image as a point.
replicate_unit = 'droplet'
min_replicates = 1


def parse_stack(stack):
    """'DPhPC_01PEG_D01_I02' -> ('DPhPC_01PEG', 'D01', 'I02')

    An optional leading date field (e.g. '260814_DPhPC_...') is dropped.
    """
    parts = stack.split('_')
    if parts[0].isdigit():
        parts = parts[1:]
    return '_'.join(parts[:-2]), parts[-2], parts[-1]


def load_and_process(boundary_file, inside_file):
    bp = pd.read_csv(os.path.join(input_folder, boundary_file))
    ip = pd.read_csv(os.path.join(input_folder, inside_file))
    df = bp.merge(ip, on=['stack', 'frame', 'droplet_id'],
                  suffixes=('_bp', '_ip'), validate='1:1')

    df[metric] = (df[boundary_col] - df[background_col]) / df[inside_col]

    parsed = df['stack'].apply(parse_stack)
    df['condition'] = [c for c, _, _ in parsed]
    df['droplet'] = [d for _, d, _ in parsed]
    df['image'] = [i for _, _, i in parsed]
    df['replicate'] = df['droplet'] if replicate_unit == 'droplet' \
        else df['droplet'] + '_' + df['image']

    df = df[['condition', 'replicate', metric]].dropna()
    # average within replicate first, so a droplet imaged twice counts once
    df = df.groupby(['condition', 'replicate'], as_index=False)[metric].mean()
    return df


def run_stats(data, xcol, ycol, order):
    """One-way ANOVA + Holm-corrected pairwise t-tests over conditions with n>=2."""
    grp_data = {name: grp[ycol].values for name, grp in data.groupby(xcol)
                if len(grp) >= 2 and name in order}
    rows = []
    if len(grp_data) < 2:
        print("Fewer than 2 conditions have n>=2 replicates - no tests run.")
        return pd.DataFrame(rows), {}

    f_stat, anova_p = stats.f_oneway(*grp_data.values())
    print(f"ANOVA F={f_stat:.2f}, p={anova_p:.4e}  (groups with n>=2: {len(grp_data)})")
    rows.append(dict(comparison='one-way ANOVA', statistic=f_stat,
                     p_raw=anova_p, p_holm=np.nan))

    pair_keys = [(a, b) for a, b in combinations(order, 2)
                 if a in grp_data and b in grp_data]
    if not pair_keys:
        return pd.DataFrame(rows), {}

    raw_pvals = [stats.ttest_ind(grp_data[a], grp_data[b])[1] for a, b in pair_keys]
    _, corrected_pvals, _, _ = multipletests(raw_pvals, method='holm')
    p_lookup = {}
    for (a, b), p_raw, p_corr in zip(pair_keys, raw_pvals, corrected_pvals):
        print(f"  {a} vs {b}: p_raw={p_raw:.4f}  p_holm={p_corr:.4f}")
        rows.append(dict(comparison=f'{a} vs {b}', statistic=np.nan,
                         p_raw=p_raw, p_holm=p_corr))
        p_lookup[(a, b)] = p_corr
        p_lookup[(b, a)] = p_corr
    return pd.DataFrame(rows), p_lookup


def p_to_stars(p):
    if p < 0.0001:
        return '****'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def bardotplot(data, xcol, ycol, order, palette, ax,
               xlabel='', ylabel='', dot_size=5, cap_size=0.15, cap_width=1):
    """Bar (mean + 95% CI) with the raw replicate points overlaid."""
    if sns_v12:
        sns.barplot(data=data, x=xcol, y=ycol, hue=xcol, palette=palette,
                    capsize=cap_size, err_kws={'linewidth': cap_width},
                    ax=ax, order=order, hue_order=order, legend=False,
                    edgecolor='white')
    else:
        sns.barplot(data=data, x=xcol, y=ycol, palette=palette, ci=95,
                    capsize=cap_size, errwidth=cap_width,
                    ax=ax, order=order, edgecolor='white')
    # Lighten the bar fills so the same-hue replicate dots stay visible on top of
    # them (a saturated dot on a saturated bar reads as an empty white ring).
    for patch in ax.patches:
        fc = patch.get_facecolor()
        patch.set_facecolor((fc[0], fc[1], fc[2], 0.40))
        patch.set_edgecolor((fc[0], fc[1], fc[2], 1.0))
        patch.set_linewidth(0.8)
    if sns_v12:
        sns.stripplot(data=data, x=xcol, y=ycol, hue=xcol, palette=palette, ax=ax,
                      edgecolor='#fff', linewidth=1, s=dot_size,
                      order=order, hue_order=order, legend=False, zorder=3)
    else:
        sns.stripplot(data=data, x=xcol, y=ycol, palette=palette, ax=ax,
                      edgecolor='#fff', linewidth=1, size=dot_size,
                      order=order, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def draw_reference_brackets(ax, order, p_lookup, y_start, y_step):
    """Bracket every condition against reference_condition, low bars first."""
    if reference_condition not in order:
        print(f"Reference condition {reference_condition} absent - no brackets drawn.")
        return
    ref_i = order.index(reference_condition)
    targets = [c for c in order if c != reference_condition and (reference_condition, c) in p_lookup]
    targets.sort(key=lambda c: abs(order.index(c) - ref_i))
    bracket_h = y_step * 0.20
    for level, c in enumerate(targets):
        x_a, x_b = sorted([ref_i, order.index(c)])
        y = y_start + level * y_step
        ax.plot([x_a, x_a, x_b, x_b],
                [y - bracket_h, y, y, y - bracket_h],
                color='black', linewidth=0.5, clip_on=False)
        ax.text((x_a + x_b) / 2, y + bracket_h * 0.3,
                p_to_stars(p_lookup[(reference_condition, c)]),
                ha='center', va='bottom', fontsize=6, clip_on=False)


bp_488 = load_and_process(boundary_file, inside_file)
print(f"{channel}: {len(bp_488)} replicate-averaged points "
      f"(replicate unit = {replicate_unit})")
counts = bp_488.groupby('condition')[metric].agg(['mean', 'std', 'count'])
print(counts)

missing = set(bp_488['condition']) - set(condition_order)
if missing:
    print(f"Warning: conditions not in condition_order and therefore not plotted: {missing}")

# Only plot conditions actually present, so a pruned dataset leaves no empty slots.
present = [c for c in condition_order
           if c in counts.index and counts.loc[c, 'count'] >= min_replicates]
absent = [c for c in condition_order if c not in present]
if absent:
    print(f"Not present in this dataset, omitted from x-axis: {absent}")
singles = [c for c in present if counts.loc[c, 'count'] == 1]
if singles:
    print(f"Single-replicate conditions (bar has no error bar): {singles}")

plt.rc('font', family='arial', weight='normal', size=8)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm = 1 / 2.54

subset = bp_488[bp_488['condition'].isin(present)]
stats_df, p_lookup = run_stats(subset, 'condition', metric, present)

n_brackets = sum(1 for c in present
                 if c != reference_condition and (reference_condition, c) in p_lookup)
data_max = subset[metric].max()
y_step = data_max * 0.11
y_start = data_max * 1.12
y_upper = y_start + max(n_brackets - 1, 0) * y_step + y_step

fig, ax = plt.subplots(figsize=(9 * cm, 8 * cm))
ax.set_ylim(0, y_upper)

bardotplot(subset, 'condition', metric, present, 'Set2', ax, ylabel=ylabel_text)
draw_reference_brackets(ax, present, p_lookup, y_start, y_step)

# ratio = 1 means boundary and interior are equally bright
ax.axhline(1.0, color='0.6', lw=0.8, ls=':', zorder=0)

for i, c in enumerate(present):
    ax.annotate(f"n={int(counts.loc[c, 'count'])}", xy=(i, 0),
                xytext=(0, 2), textcoords='offset points',
                ha='center', va='bottom', fontsize=6, color='0.3')

ax.set_xticks(range(len(present)))
ax.set_xticklabels([condition_labels.get(c, c) for c in present])
ax.tick_params(axis='x', rotation=45)
for label in ax.get_xticklabels():
    label.set_ha('right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

vc = subset['condition'].value_counts()
if (vc >= 2).sum() >= 2:
    multi = subset[subset['condition'].isin(vc[vc >= 2].index)]
    tukey = pairwise_tukeyhsd(multi[metric], multi['condition'], alpha=0.05)
    print(f"\nTukey HSD (conditions with n>=2 only):\n{tukey}")
else:
    print("\nTukey HSD skipped: fewer than 2 conditions with n>=2.")

output_path = os.path.join(output_folder, f'boundary_{metric}_{channel}.svg')
plt.savefig(output_path, bbox_inches='tight', dpi=300, format='svg')
plt.savefig(output_path.replace('.svg', '.png'), bbox_inches='tight', dpi=300)
if len(stats_df):
    stats_path = os.path.join(input_folder, f'boundary_{metric}_{channel}_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"Stats: {stats_path}")
print(f"\nSaved to {output_path}")