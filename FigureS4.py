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

# ============================================================================
# PATHS
# ============================================================================
BAR_DIR = r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\Diffusion concentration\Cleaned_excel_table"
OUTPUT_DIR = r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\SVG_for_final_figures"

# PEG supplementary data
PEG_BASE_DIR = r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\Supplementary paper"

# NEW: three PEG conditions pulled from the supplementary folders.
# - 'well_withPEG'       -> 8-well +PEG data (the "no PEG" 8-well folders are
#                            intentionally NOT included here; the existing
#                            'well' / Chamber condition from BAR_DIR already
#                            covers the no-PEG baseline).
# - 'coverslip_withPEG'  -> Coverslip +PEG data
# - 'coverslip_noPEG'    -> Coverslip no-PEG data
PEG_GROUPS = {
    'coverslip_withPEG': [
        (os.path.join('260907_PEG', 'Coverslip_PEG'), '260907'),
        (os.path.join('260903_PEG', 'Peg_coverslip'), '260903'),
    ],
    'coverslip_noPEG': [
        (os.path.join('260907_PEG', 'Coverslip_No_PEG'), '260907'),
        (os.path.join('260903_PEG', 'No_PEG_coverslip'), '260903'),
    ],
}

# ============================================================================
# LOAD BAR DATA — well vs DIB-BOT (existing conditions)
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
    df = pd.read_excel(fp)
    filename = os.path.basename(fp).lower()

    if 'dendra' in filename:
        molecule = 'dendra'
    elif 'gfp' in filename:
        molecule = 'GFP'
    elif 'mcherry_lys' in filename:
        molecule = 'mcherry_lys'
    elif 'mcherry' in filename:
        molecule = 'mcherry'
    else:
        molecule = 'unknown'

    condition = 'well' if 'well' in filename else ('dibbot' if 'dibbot' in filename else 'unknown')

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

# ============================================================================
# LOAD BAR DATA — PEG supplementary conditions (well_withPEG, coverslip_withPEG,
# coverslip_noPEG), only using the "_table" excel files, mCherry + dendra only.
# ============================================================================
def _list_table_excels(folder):
    files = []
    for f in os.listdir(folder):
        if not f.lower().endswith(('.xlsx', '.xls')):
            continue
        if f.startswith('~$'):
            continue
        low = f.lower()
        if '_table' not in low:
            continue
        if '_count' in low or '_curve' in low:
            continue
        if 'srb' in low:
            continue
        files.append(os.path.join(folder, f))
    return files

def get_molecule(filename):
    low = filename.lower()
    if 'dendra' in low:
        return 'dendra'
    elif 'mcherry' in low:
        return 'mcherry'
    else:
        return 'unknown'

all_dfs_peg = []
for group, folder_list in PEG_GROUPS.items():
    for rel_folder, biorep in folder_list:
        full_folder = os.path.join(PEG_BASE_DIR, rel_folder)
        if not os.path.isdir(full_folder):
            print(f"WARNING: folder not found, skipping: {full_folder}")
            continue
        for fp in _list_table_excels(full_folder):
            df = pd.read_excel(fp)
            molecule = get_molecule(os.path.basename(fp))
            if molecule == 'unknown':
                continue
            df['Sample'] = f"{molecule}_{group}"
            df['Molecule'] = molecule
            df['Condition'] = group
            df['BioRep'] = biorep
            all_dfs_peg.append(df)

KEEP_COLS = ['Diffusion Coefficient 1 µm²/s', 'Concentration nM',
             'Sample', 'Molecule', 'Condition', 'BioRep']

if all_dfs_peg:
    combined_df_peg = pd.concat(all_dfs_peg, ignore_index=True)
    plot_df_peg = combined_df_peg[
        (combined_df_peg['Molecule'] != 'unknown') &
        (combined_df_peg['Diffusion Coefficient 1 µm²/s'] >= 5)
    ].copy()

    # Merge all PEG condition rows into the main plotting dataframe
    plot_df = pd.concat(
        [plot_df[KEEP_COLS], plot_df_peg[KEEP_COLS]],
        ignore_index=True
    )
else:
    print("WARNING: no PEG supplementary files found — check PEG_BASE_DIR / PEG_GROUPS paths.")

# ============================================================================
# HELPERS
# ============================================================================
molecule_order  = ['dendra', 'mcherry']

# NEW: 5 conditions total — well/dibbot from the original dataset, plus the
# three PEG supplementary conditions.
condition_order = ['well', 'dibbot', 'coverslip_withPEG', 'coverslip_noPEG']

palette = {
    'well':               '#3d9973',   # green
    'dibbot':             '#0496c7',   # blue
    'coverslip_withPEG':  '#C97064',   # coral
    'coverslip_noPEG':    '#8E6BAF',   # purple
}

condition_labels = {
    'dibbot': 'DIB-BOT',
    'well': 'Chamber',
    'coverslip_noPEG': 'Coverslip',
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
# FIGURE LAYOUT — bar charts only, no FCS curve panels
# ============================================================================
fig_width  = 18 * cm
fig_height = 9 * cm

fig = plt.figure(figsize=(fig_width, fig_height))

gs = gridspec.GridSpec(1, 2, figure=fig,
                        left=0.08, right=0.98,
                        top=0.90, bottom=0.12,
                        wspace=0.45)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

# ============================================================================
# BAR PANELS — sns.barplot + sns.stripplot (5 conditions per molecule)
# ============================================================================
def draw_bar_panel(ax, col, ymax, label=None):
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

    # Statistical annotations — DIB-BOT compared against every other condition.
    n_hue     = len(condition_order)
    bar_width = 0.8 / n_hue
    
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
            bracket_gap = ymax * 0.10   # minimum vertical gap between stacked brackets

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
    ax.set_ylabel(label if label is not None else col)
    ax.set_xlabel('')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', which='both', direction='out', length=3, width=0.8, left=True)
    ax.tick_params(axis='x', which='both', length=0, bottom=False)
    ax.legend('', frameon=False)

# NOTE: with 5 conditions there are up to 4 stacked brackets per molecule now
# (vs. 2 before) — ymax has been raised to give the brackets room. Adjust if
# your data still clips.
draw_bar_panel(ax_a, 'Diffusion Coefficient 1 µm²/s', 350, label='Diffusion Coefficient (µm²/s)')

draw_bar_panel(ax_b, 'Concentration nM', 1100, label='Concentration (nM)')

# Legend on ax_a only
legend_handles = [
    mpatches.Patch(color=palette[cond], label=condition_labels[cond])
    for cond in condition_order
]
ax_a.legend(handles=legend_handles, frameon=False, fontsize=7, ncol=1, loc='upper right')

# ---- Panel labels ----
for ax, label in zip([ax_a, ax_b], ['A', 'B']):
    ax.text(-0.15, 1.05, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')

plt.show()
# %%
out_path = os.path.join(OUTPUT_DIR, "Figure_3_Supp_PEG_No_chamberPEG.svg")
fig.savefig(out_path, format="svg", dpi=300)
print(f"Saved to: {out_path}")