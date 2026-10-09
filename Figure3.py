# %%
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
from scipy import stats
from collections import defaultdict
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
RAW_REPLICATE_INDEX = 3

# ============================================================================
# PATHS
# ============================================================================
BAR_DIR = r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\Diffusion concentration\Plotting_Figure_3"
OUTPUT_DIR = r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\SVG_for_final_figures"

RAW_FILES = {
    'dendra': r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\Supplementary paper\260319_dendra_dibbot.xlsx",
    'mcherry': r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\Supplementary paper\260319_mcherry_dibbot.xlsx",
}

# ============================================================================
# LOAD RAW DATA (single replicate: both raw correlation and fit curve)
# ============================================================================
raw_data = {}
for cond, fp in RAW_FILES.items():
    df_raw = pd.read_excel(fp, header=1)
    col_start = RAW_REPLICATE_INDEX * 5

    df_corr = df_raw.iloc[:, [col_start, col_start + 1]].copy()
    df_corr.columns = ['Time [ms]', 'Correlation Channel 1']
    df_corr = df_corr.dropna()
    df_corr = df_corr[pd.to_numeric(df_corr['Time [ms]'], errors='coerce').notna()]
    df_corr = df_corr.astype({'Time [ms]': float, 'Correlation Channel 1': float})

    plateau = df_corr['Correlation Channel 1'].iloc[:10].mean()
    if plateau != 0:
        df_corr['Correlation Channel 1'] = df_corr['Correlation Channel 1'] / plateau

    df_fit = df_raw.iloc[:, [col_start + 3, col_start + 4]].copy()
    df_fit.columns = ['Time [ms]', 'Fit Channel 1']
    df_fit = df_fit.dropna()
    df_fit = df_fit[pd.to_numeric(df_fit['Time [ms]'], errors='coerce').notna()]
    df_fit = df_fit.astype({'Time [ms]': float, 'Fit Channel 1': float})

    plateau_fit = df_fit['Fit Channel 1'].iloc[:10].mean()
    if plateau_fit != 0:
        df_fit['Fit Channel 1'] = df_fit['Fit Channel 1'] / plateau_fit

    raw_data[cond] = {'corr': df_corr, 'fit': df_fit}

# ============================================================================
# LOAD BAR DATA — well vs DIB-BOT
# ============================================================================
def _list_excels(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.xlsx', '.xls')) and not f.startswith('~$')
    ]

def extract_biorep(filename):
    return os.path.basename(filename).split('_')[0]

all_dfs = []
for fp in _list_excels(BAR_DIR):
    filename = os.path.basename(fp).lower()

    # Molecule: only dendra and mcherry are used in this figure
    if 'dendra' in filename:
        molecule = 'dendra'
    elif 'mcherry_lys' in filename or 'gfp' in filename:
        continue  # not part of this figure
    elif 'mcherry' in filename:
        molecule = 'mcherry'
    else:
        print(f"WARNING: unrecognised molecule, skipping: {os.path.basename(fp)}")
        continue

    # Condition: 'chamber' is stored internally as 'chamber' (palette/order keys)
    if 'chamber' in filename:
        condition = 'chamber'
    elif 'dibbot' in filename:
        condition = 'dibbot'
    elif 'coverslippeg' in filename:
        condition = 'coverslip_withPEG'   # matches existing palette/order keys
    else:
        print(f"WARNING: unrecognised condition, skipping: {os.path.basename(fp)}")
        continue

    df = pd.read_excel(fp)
    df['Sample'] = f"{molecule}_{condition}"
    df['Molecule'] = molecule
    df['Condition'] = condition
    df['BioRep'] = extract_biorep(filename)
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)
plot_df = combined_df[
    (combined_df['Sample'] != 'unknown_unknown') &
    (combined_df['Diffusion Coefficient 1 µm²/s'] >= 5)
].copy()

#=============================================================================
# HELPERS
# ============================================================================
colors  = {'dendra': '#9B59B6', 'mcherry': '#E7549E'}

# NEW: teal sits visually "between" your existing green (chamber) and blue (dibbot)
palette = {
    'chamber': '#3d9973',
    'dibbot': '#0496c7',
    'coverslip_withPEG': '#C97064',
}

molecule_order   = ['dendra', 'mcherry']
condition_order  = ['chamber', 'dibbot', 'coverslip_withPEG']   # NEW: third condition added
condition_labels = {
    'chamber': 'Chamber',
    'dibbot': 'DIB-BOT',
    'coverslip_withPEG': 'Coverslip\n+PEG',
}

def get_sig_label(p):
    if p < 0.0001:
        return '****'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

# ============================================================================
# FIGURE LAYOUT
# ============================================================================
fig_width  = 18 * cm
fig_height = 15 * cm

fig = plt.figure(figsize=(fig_width, fig_height))

gs_top = gridspec.GridSpec(1, 3, figure=fig,
                            width_ratios=[8, 5, 5],
                            left=0.08, right=0.98,
                            top=0.95, bottom=0.60,
                            wspace=0.2)

gs_bot = gridspec.GridSpec(1, 2, figure=fig,
                            left=0.08, right=0.98,
                            top=0.45, bottom=0.05,
                            wspace=0.45)

ax_a         = fig.add_subplot(gs_top[0, 0])
ax_b_dendra  = fig.add_subplot(gs_top[0, 1])
ax_b_mcherry = fig.add_subplot(gs_top[0, 2], sharey=ax_b_dendra)
ax_c         = fig.add_subplot(gs_bot[0, 0])
ax_d         = fig.add_subplot(gs_bot[0, 1])

ax_a.axis('off')

# ============================================================================
# FCS PANELS — single replicate fit + raw
# ============================================================================
def draw_fcs_panel(ax, cond):
    data    = raw_data[cond]
    df_corr = data['corr']
    df_fit  = data['fit']

    ax.plot(
        df_corr['Time [ms]'],
        df_corr['Correlation Channel 1'],
        color=colors[cond],
        linestyle='-',
        linewidth=1.0,
        alpha=0.8,
        zorder=4,
    )

    ax.plot(
        df_fit['Time [ms]'],
        df_fit['Fit Channel 1'],
        color='black',
        linestyle='--',
        linewidth=1.5,
        alpha=0.9,
        zorder=5,
    )

    ax.set_xscale('log')
    ax.set_ylim(0, 1.14)
    ax.set_xlabel('Time [ms]')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis='both', which='both', direction='out', length=3, width=0.8, bottom=True, left=True)
    ax.set_xticks([1e-2, 1e0, 1e2])
    ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
    ax.tick_params(axis='x', which='minor', length=0)
    ax.tick_params(axis='x', which='major', direction='out', length=3, width=0.8, bottom=True)

    labels = {'dendra': 'GST-Dendra2', 'mcherry': 'mCherry'}
    display_name = labels.get(cond, cond.capitalize())
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label=f'{display_name} fit'),
        matplotlib.lines.Line2D([0], [0], color=colors[cond], linewidth=1.0, linestyle='-',  label=f'{display_name} raw'),
    ]
    ax.legend(handles=legend_elements, title=None, frameon=False, fontsize=8)

draw_fcs_panel(ax_b_dendra, 'dendra')
ax_b_dendra.set_ylabel('Normalised G(t)')

draw_fcs_panel(ax_b_mcherry, 'mcherry')
ax_b_mcherry.set_ylabel('')
plt.setp(ax_b_mcherry.get_yticklabels(), visible=False)

# ============================================================================
# BAR PANELS — sns.barplot + sns.stripplot (now 3 conditions per molecule)
# ============================================================================
def draw_bar_panel(ax, col, ymax):
    ax.set_ylim(0, ymax)

    sns.barplot(
        data=plot_df,
        x='Molecule',
        y=col,
        hue='Condition',
        palette=palette,
        capsize=0.4,
        err_kws={'linewidth': 1},
        ax=ax,
        dodge=True,
        order=molecule_order,
        hue_order=condition_order,
        edgecolor='white',
        errorbar='se',
    )

    sns.stripplot(
        data=plot_df,
        x='Molecule',
        y=col,
        hue='Condition',
        palette=palette,
        ax=ax,
        edgecolor='white',
        linewidth=0.8,
        s=6,
        order=molecule_order,
        hue_order=condition_order,
        dodge=True,
    )

    # Statistical annotations — now pairwise across all 3 conditions per molecule
    n_hue     = len(condition_order)
    bar_width = 0.8 / n_hue

    # Bracket pairs to test/draw — DIB-BOT compared against every other condition only.
    # (cond_a, cond_b, hue_index_a, hue_index_b) — hue indices follow condition_order.
    dibbot_hue_idx = condition_order.index('dibbot')
    pair_list = [
        (cond, 'dibbot', condition_order.index(cond), dibbot_hue_idx)
        for cond in condition_order if cond != 'dibbot'
    ]

    for i, molecule in enumerate(molecule_order):
        subset = plot_df[plot_df['Molecule'] == molecule]
        grp_data = {
            cond: subset[subset['Condition'] == cond][col].dropna().values
            for cond in condition_order
        }

        # Track the y-position of the highest bracket drawn so far for this
        # molecule, so each new bracket gets a guaranteed minimum gap above
        # the last one — regardless of how close the underlying data maxes are.
        current_top = None

        for cond_a, cond_b, hue_a, hue_b in pair_list:
            vals1 = grp_data.get(cond_a, np.array([]))
            vals2 = grp_data.get(cond_b, np.array([]))

            if len(vals1) < 2 or len(vals2) < 2:
                continue

            _, p_levene = stats.levene(vals1, vals2)
            equal_var = p_levene > 0.05
            _, p_val = stats.ttest_ind(vals1, vals2, equal_var=equal_var)
            label = get_sig_label(p_val)

            print(f"{molecule} | {col} — {cond_a} vs {cond_b} "
                  f"({'Student' if equal_var else 'Welch'}): p={p_val:.4f} ({label})")

            if label == 'ns':
                continue

            x_a = i - 0.4 + bar_width * (hue_a + 0.5)
            x_b = i - 0.4 + bar_width * (hue_b + 0.5)
            y_max_data = max(vals1.max(), vals2.max())
            bracket_h  = ymax * 0.01
            bracket_gap = ymax * 0.12   # minimum vertical gap between stacked brackets

            candidate_y = y_max_data * 1.1
            if current_top is None:
                y_bracket = candidate_y
            else:
                y_bracket = max(candidate_y, current_top + bracket_gap)

            ax.plot([x_a, x_a, x_b, x_b],
                    [y_bracket - bracket_h, y_bracket, y_bracket, y_bracket - bracket_h],
                    color='black', linewidth=1.5, clip_on=False)
            ax.text((x_a + x_b) / 2, y_bracket + bracket_h * 0.3, label,
                    ha='center', va='bottom', fontsize=15, clip_on=False)

            current_top = y_bracket

    ax.set_xticklabels(['GST-Dendra2', 'mCherry'])
    ax.set_ylabel(col)
    ax.set_xlabel('')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', which='both', direction='out', length=3, width=0.8, left=True)
    ax.tick_params(axis='x', which='both', length=0, bottom=False)
    ax.legend('', frameon=False)

# NEW: extra headroom for stacked significance brackets with 3 conditions
draw_bar_panel(ax_c, 'Diffusion Coefficient 1 µm²/s', 380)
draw_bar_panel(ax_d, 'Concentration nM', 750)

# Legend on ax_c only
legend_handles = [
    mpatches.Patch(color=palette['chamber'],   label=condition_labels['chamber']),
    mpatches.Patch(color=palette['dibbot'], label=condition_labels['dibbot']),
    mpatches.Patch(color=palette['coverslip_withPEG'], label='Coverslip +PEG'),
]
ax_c.legend(handles=legend_handles, frameon=False, fontsize=8)

# ---- Panel labels ----
for ax, label in zip([ax_a, ax_b_dendra, ax_b_mcherry, ax_c, ax_d], ['A', 'B', 'C', 'D', 'E']):
    ax.text(-0.15, 1.05, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')

plt.show()
# %%
out_path = os.path.join(OUTPUT_DIR, "Figure_3_with_coverslip.svg")
fig.savefig(out_path, format="svg", dpi=300)
print(f"Saved to: {out_path}")

