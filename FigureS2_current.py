# Figure S2: PEG-lipid coating, 488 nm. EXPAL266 and EXPAL266_2 pooled.
# Panel A: one representative confocal section per condition, each normalised to its
#          own background-subtracted interior mean, common 0-vmax greyscale.
# Panel B: boundary / interior ratio, bar + replicate dots + stats, both runs pooled.
#
# Metric: boundary_mean_intensity_bgsub / mean_intensity_bgsub
# Stats:  one-way ANOVA, then Dunnett's test of each condition against oil only.
#
# Reads  {base}{folder}/Results/boundary_pixels_488.csv + inside_pixels_488.csv
#        {base}{folder}/Extracted_channels/{stack}_488.tif
# Writes {base}Figures/EXPAL266_Fig_PEG_coating.svg/.png/.pdf
#        {base}Figures/EXPAL266_Fig_PEG_coating_panelA_sources.csv
#        {base}Figures/boundary_coating_ratio_488_stats.csv

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import tifffile
from packaging.version import Version

# --- paths -------------------------------------------------------------------
base_path = '/Users/aleksalakic/Desktop/supp_figure_s2_copy/'
# One entry per experiment; replicates from all roots are pooled in panel B.
root_folders = [f'{base_path}EXPAL266_analysis/',
                f'{base_path}EXPAL266_2_analysis/']
input_subdir = 'Results/'
channels_subdir = 'Extracted_channels/'
output_folder = f'{base_path}Figures/'
os.makedirs(output_folder, exist_ok=True)

stem = 'EXPAL266_Fig_PEG_coating'

# --- analysis config ---------------------------------------------------------
channel = '488'
boundary_file = f'boundary_pixels_{channel}.csv'
inside_file = f'inside_pixels_{channel}.csv'

metric = 'coating_ratio'
ylabel_text = 'Boundary / interior intensity'

# Both writers subtract the same per-frame background, so the bgsub columns are
# divided directly.
boundary_col = 'boundary_mean_intensity_bgsub'
inside_col = 'mean_intensity_bgsub'

condition_order = ['oil', 'DPhPC', 'DPhPC_01PEG', 'DPhPC_05PEG',
                   'DPhPC_1PEG', 'DPhPC_5PEG', 'DPhPC_10PEG']

# 01PEG / 05PEG are 0.1% / 0.5% (zero-padded decimals)
condition_labels = {'oil': 'oil only', 'DPhPC': '0% PEG',
                    'DPhPC_01PEG': '0.1% PEG', 'DPhPC_05PEG': '0.5% PEG',
                    'DPhPC_1PEG': '1% PEG', 'DPhPC_5PEG': '5% PEG',
                    'DPhPC_10PEG': '10% PEG'}

# Tile borders, matched to the Set2 bar colours in panel B.
condition_colors = {'oil': '#66c2a5', 'DPhPC': '#fc8d62', 'DPhPC_01PEG': '#8da0cb',
                    'DPhPC_05PEG': '#e78ac3', 'DPhPC_1PEG': '#a6d854',
                    'DPhPC_5PEG': '#ffd92f', 'DPhPC_10PEG': '#e5c494'}

# Dunnett control. Every other condition is tested against this one only, so the
# correction covers exactly the comparisons that are drawn.
reference_condition = 'oil'

# 'droplet' pools I01/I02 of the same D-number, 'image' treats each image as a point.
replicate_unit = 'droplet'
min_replicates = 1

# --- panel A config ----------------------------------------------------------
# Representative droplet per condition; the root is resolved from the data.
picks = {'oil': 'oil_D03_I01',
         'DPhPC': 'DPhPC_D01_I02',
         'DPhPC_01PEG': 'DPhPC_01PEG_D02_I01',
         'DPhPC_05PEG': 'DPhPC_05PEG_D03_I01',
         'DPhPC_1PEG': 'DPhPC_1PEG_D01_I01',
         'DPhPC_5PEG': 'DPhPC_5PEG_D01_I01',
         'DPhPC_10PEG': 'DPhPC_10PEG_D03_I01'}

# Tiles come from this root (0 = EXPAL266) when a stack name exists in both.
picks_root_index = 0

px_um = 1.1220007455875922   # Leica .lif metadata, um/px
scalebar_um = 100.0
vmax = 2.5                   # top of the shared display scale, x interior mean

# --- layout, inches (fig width 7.09 in = 180 mm) ------------------------------
fig_w = 7.09
left_m, right_m = 0.24, 0.34
tile_gap = 0.045
cbar_gap, cbar_w = 0.10, 0.085
top_m, panel_gap, bottom_m = 0.30, 0.75, 0.55
panel_b_w, panel_b_h = 3.30, 2.90

# seaborn changed the categorical API in 0.12 (err_kws/legend vs errwidth/size)
sns_v12 = Version(sns.__version__) >= Version('0.12')


def parse_stack(stack):
    """'DPhPC_01PEG_D01_I02' -> ('DPhPC_01PEG', 'D01', 'I02')

    An optional leading date field (e.g. '260814_DPhPC_...') is dropped.
    """
    parts = stack.split('_')
    if parts[0].isdigit():
        parts = parts[1:]
    return '_'.join(parts[:-2]), parts[-2], parts[-1]


def load_experiment(root, exp_index):
    """Per-droplet metric table for one root, tagged with its experiment index."""
    bp = pd.read_csv(os.path.join(root, input_subdir, boundary_file))
    ip = pd.read_csv(os.path.join(root, input_subdir, inside_file))
    df = bp.merge(ip, on=['stack', 'frame', 'droplet_id'],
                  suffixes=('_bp', '_ip'), validate='1:1')

    df[metric] = df[boundary_col] / df[inside_col]

    parsed = df['stack'].apply(parse_stack)
    df['condition'] = [c for c, _, _ in parsed]
    df['droplet'] = [d for _, d, _ in parsed]
    df['image'] = [i for _, _, i in parsed]

    unit = df['droplet'] if replicate_unit == 'droplet' \
        else df['droplet'] + '_' + df['image']
    # experiment index keeps D-numbers from colliding between experiments
    df['experiment'] = exp_index
    df['replicate'] = f'E{exp_index}_' + unit
    return df[['experiment', 'stack', 'condition', 'replicate', metric]].dropna()


def load_normalisation(root):
    """Per-stack background and interior mean, for the panel A display scaling."""
    bp = pd.read_csv(os.path.join(root, input_subdir, boundary_file))
    ip = pd.read_csv(os.path.join(root, input_subdir, inside_file))
    bg = bp.groupby('stack')['background'].mean()
    plateau = ip.groupby('stack')[inside_col].mean()
    return pd.concat([bg, plateau], axis=1)


def resolve_picks(per_droplet):
    """Map each condition to (experiment index, stack, mean metric).

    A stack present in both roots is taken from picks_root_index. The metric is
    kept for the source table only; it is no longer drawn on the tiles.
    """
    stack_ratio = per_droplet.groupby(['experiment', 'stack'])[metric].mean()
    resolved = {}
    for key, stack in picks.items():
        hits = [e for e, s in stack_ratio.index if s == stack]
        if not hits:
            available = sorted(per_droplet.loc[per_droplet['condition'] == key,
                                               'stack'].unique())
            raise KeyError(f'{stack} not found in any root. '
                           f'Stacks for {key}: {available}')
        exp = picks_root_index if picks_root_index in hits else hits[0]
        if len(hits) > 1:
            print(f'{stack} present in roots {hits}; using {exp}.')
        resolved[key] = (exp, stack, float(stack_ratio.loc[(exp, stack)]))
    return resolved


def load_display_images(resolved):
    """(raw - background) / interior_mean_bgsub for each representative image.

    1.0 is the droplet's own interior level, so the greyscale is comparable
    between tiles. Both constants are read from the existing metrics tables.
    """
    norms = {i: load_normalisation(root) for i, root in enumerate(root_folders)}
    images = {}
    for key, (exp, stack, _) in resolved.items():
        path = os.path.join(root_folders[exp], channels_subdir,
                            f'{stack}_{channel}.tif')
        raw = tifffile.imread(path).astype(np.float32)
        if raw.ndim == 3:
            raw = raw[0]
        bg = norms[exp].loc[stack, 'background']
        plateau = norms[exp].loc[stack, inside_col]
        images[key] = (raw - bg) / plateau
    return images


def run_stats(data, xcol, ycol, order):
    """One-way ANOVA, then Dunnett's test of each condition against the control.

    Dunnett corrects for the k-1 control comparisons only and accounts for the
    shared control group, so it is less conservative here than all-pairwise
    correction over k(k-1)/2 tests.
    """
    grp_data = {name: grp[ycol].values for name, grp in data.groupby(xcol)
                if len(grp) >= 2 and name in order}
    rows = []
    if reference_condition not in grp_data or len(grp_data) < 2:
        print(f'Control {reference_condition} absent or fewer than 2 conditions '
              f'with n>=2 - no tests run.')
        return pd.DataFrame(rows), {}

    f_stat, anova_p = stats.f_oneway(*grp_data.values())
    print(f'ANOVA F={f_stat:.2f}, p={anova_p:.4e}  (groups with n>=2: {len(grp_data)})')
    rows.append(dict(comparison='one-way ANOVA', statistic=f_stat, p_value=anova_p))

    targets = [c for c in order if c in grp_data and c != reference_condition]
    result = stats.dunnett(*[grp_data[c] for c in targets],
                           control=grp_data[reference_condition])

    p_lookup = {}
    for c, t_stat, p in zip(targets, result.statistic, result.pvalue):
        print(f'  {reference_condition} vs {c}: t={t_stat:.2f}  p_dunnett={p:.4f}')
        rows.append(dict(comparison=f'{reference_condition} vs {c}',
                         statistic=t_stat, p_value=p))
        p_lookup[(reference_condition, c)] = p
        p_lookup[(c, reference_condition)] = p
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
    # Lighten the fills so same-hue dots stay visible on top of the bars.
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
    """Bracket every condition against reference_condition, nearest bars first."""
    if reference_condition not in order:
        print(f'Reference condition {reference_condition} absent - no brackets drawn.')
        return
    ref_i = order.index(reference_condition)
    targets = [c for c in order
               if c != reference_condition and (reference_condition, c) in p_lookup]
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


def draw_panel_a(fig, images, tile, row_w, y_bottom, fig_h):
    """Image tile row plus the shared colourbar. Returns the tile axes."""
    axes = []
    handle = None
    for i, key in enumerate(condition_order):
        ax = fig.add_axes([(left_m + i * (tile + tile_gap)) / fig_w, y_bottom,
                           tile / fig_w, tile / fig_h])
        handle = ax.imshow(images[key], cmap='gray', vmin=0, vmax=vmax,
                           interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(condition_colors[key])
            spine.set_linewidth(1.5)
        ax.set_title(condition_labels[key], fontsize=7, pad=3, color='k')
        axes.append(ax)

    bar_px = scalebar_um / px_um
    ax_last = axes[-1]
    n_px = images[condition_order[-1]].shape[1]
    x1 = n_px - 30
    ax_last.plot([x1 - bar_px, x1], [n_px - 34] * 2,
                 color='w', lw=2.4, solid_capstyle='butt')
    ax_last.text(x1, n_px - 64, f'{scalebar_um:.0f} \u00b5m',
                 color='w', fontsize=6, ha='right', va='bottom')

    cax = fig.add_axes([(left_m + row_w + cbar_gap) / fig_w, y_bottom,
                        cbar_w / fig_w, tile / fig_h])
    cbar = fig.colorbar(handle, cax=cax)
    cbar.set_ticks([0, 1, vmax])
    cbar.set_ticklabels(['0', '1', f'\u2265{vmax:g}'])
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_visible(False)
    cbar.set_label('Intensity /\ninterior mean', fontsize=6.5, labelpad=3)
    return axes


# --- load and pool -----------------------------------------------------------
per_droplet = pd.concat([load_experiment(root, i)
                         for i, root in enumerate(root_folders)], ignore_index=True)
print(f'{channel}: {len(per_droplet)} droplets from {len(root_folders)} experiment(s)')
print(per_droplet.groupby(['experiment', 'condition'])[metric].agg(['mean', 'count']))

# average within replicate first, so a droplet imaged twice counts once
pooled = per_droplet.groupby(['condition', 'replicate'], as_index=False)[metric].mean()
print(f'{len(pooled)} replicate-averaged points (replicate unit = {replicate_unit})')
counts = pooled.groupby('condition')[metric].agg(['mean', 'std', 'count'])
print(counts)

missing = set(pooled['condition']) - set(condition_order)
if missing:
    print(f'Warning: conditions not in condition_order, not plotted: {missing}')

present = [c for c in condition_order
           if c in counts.index and counts.loc[c, 'count'] >= min_replicates]
absent = [c for c in condition_order if c not in present]
if absent:
    print(f'Not present in this dataset, omitted from x-axis: {absent}')
singles = [c for c in present if counts.loc[c, 'count'] == 1]
if singles:
    print(f'Single-replicate conditions (bar has no error bar): {singles}')

# --- panel A inputs ----------------------------------------------------------
resolved_picks = resolve_picks(per_droplet)
images = load_display_images(resolved_picks)

# --- statistics --------------------------------------------------------------
subset = pooled[pooled['condition'].isin(present)]
stats_df, p_lookup = run_stats(subset, 'condition', metric, present)

# --- figure ------------------------------------------------------------------
plt.rc('font', family='arial', weight='normal', size=8)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42

avail = fig_w - left_m - right_m - cbar_gap - cbar_w
tile = (avail - 6 * tile_gap) / 7
row_w = 7 * tile + 6 * tile_gap
fig_h = top_m + tile + panel_gap + panel_b_h + bottom_m

fig = plt.figure(figsize=(fig_w, fig_h))
y_a = 1 - (top_m + tile) / fig_h
axes_a = draw_panel_a(fig, images, tile, row_w, y_a, fig_h)

ax_b = fig.add_axes([(left_m + (row_w - panel_b_w) / 2) / fig_w,
                     bottom_m / fig_h, panel_b_w / fig_w, panel_b_h / fig_h])

n_brackets = sum(1 for c in present
                 if c != reference_condition and (reference_condition, c) in p_lookup)
data_max = subset[metric].max()
y_step = data_max * 0.11
y_start = data_max * 1.12
y_upper = y_start + max(n_brackets - 1, 0) * y_step + y_step
ax_b.set_ylim(0, y_upper)

bardotplot(subset, 'condition', metric, present, 'Set2', ax_b, ylabel=ylabel_text)
draw_reference_brackets(ax_b, present, p_lookup, y_start, y_step)

# ratio = 1 means boundary and interior are equally bright
ax_b.axhline(1.0, color='0.6', lw=0.8, ls=':', zorder=0)

for i, c in enumerate(present):
    ax_b.annotate(f"n={int(counts.loc[c, 'count'])}", xy=(i, 0),
                  xytext=(0, 2), textcoords='offset points',
                  ha='center', va='bottom', fontsize=6, color='0.3')

ax_b.set_xticks(range(len(present)))
ax_b.set_xticklabels([condition_labels.get(c, c) for c in present])
ax_b.tick_params(axis='x', rotation=45)
for label in ax_b.get_xticklabels():
    label.set_ha('right')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

axes_a[0].text(-0.16, 1.30, 'A', transform=axes_a[0].transAxes,
               fontsize=10, fontweight='bold', va='top', ha='left')
ax_b.text(-0.16, 1.05, 'B', transform=ax_b.transAxes,
          fontsize=10, fontweight='bold', va='top', ha='left')

# --- write -------------------------------------------------------------------
fig_path = os.path.join(output_folder, f'{stem}.svg')
fig.savefig(fig_path, bbox_inches='tight', format='svg', facecolor='w')
fig.savefig(os.path.join(output_folder, f'{stem}.png'),
            bbox_inches='tight', dpi=400, facecolor='w')
fig.savefig(os.path.join(output_folder, f'{stem}.pdf'),
            bbox_inches='tight', facecolor='w')

sources = pd.DataFrame([{'panel_label': condition_labels[k],
                         'condition_key': k,
                         'root': root_folders[resolved_picks[k][0]],
                         'stack': resolved_picks[k][1],
                         'tif_file': f'{channels_subdir}{resolved_picks[k][1]}_{channel}.tif',
                         metric: round(resolved_picks[k][2], 4)}
                        for k in condition_order])
sources_path = os.path.join(output_folder, f'{stem}_panelA_sources.csv')
sources.to_csv(sources_path, index=False)

if len(stats_df):
    stats_path = os.path.join(output_folder, f'boundary_{metric}_{channel}_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f'Stats:   {stats_path}')
print(f'Sources: {sources_path}')
print(f'Saved:   {fig_path}')