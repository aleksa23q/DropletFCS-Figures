import os
import re
from pathlib import PureWindowsPath
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sns.set_theme(style="white", context="notebook")

# ============================================================================
# FIGURE SETTINGS
# ============================================================================
font = {'family': 'arial', 'weight': 'normal', 'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm = 1/2.54

def _dedupe_columns(cols):
    new_cols, seen = [], {}
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    return new_cols

def _read_one_fcs(file_path, sample_label=None):
    df = pd.read_excel(file_path, skiprows=1)
    df.columns = _dedupe_columns(df.columns)

    fit_cols = [c for c in df.columns
                if (('Fit Channel 1' in c or 'Fit Channel 2' in c or 'Fit Channel 1 -> 2' in c)
                    and 'Residuals Channel' not in c)]
    keep = ['Time [ms]'] + fit_cols
    df = df[keep]

   
    label_map = {
        'Fit Channel 1': 'Fit/Green channel 1',
        'Fit Channel 2': 'Fit/Red channel 2',
        'Fit Channel 1 -> 2': 'Fit channel 1>2',
    }

    long_parts = []
    for key in ['Fit Channel 1', 'Fit Channel 2', 'Fit Channel 1 -> 2']:
        if key == 'Fit Channel 1 -> 2':
            cols = ['Time [ms]'] + [c for c in df.columns if 'Fit Channel 1 -> 2' in c]
        else:
            cols = ['Time [ms]'] + [c for c in df.columns if (key in c and '->' not in c)]
        if len(cols) == 1:
            continue
        sub = df[cols].melt(id_vars='Time [ms]', var_name='Replicate', value_name='Value')
        sub['Channel'] = label_map[key]
        long_parts.append(sub)

    long_df = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()

    
    p = PureWindowsPath(file_path)
    m = re.search(r'(20\d{6})', str(p))
    date = m.group(1) if m else 'unknown'

    lower = str(p).lower()
    location = 'well' if 'well' in lower else 'dibbot'
    fname = p.name.lower()

    if 'dendra' in fname and 'mcherry' in fname:
        sample = 'Dendra+mCherry'
    elif 'fl1' in fname:
        sample = 'FL1'
    elif any(x in fname for x in ['ut', 'untransfected', '-vecontrol', 'negative']):
        sample = 'UT/Control'
    else:
        sample = sample_label or 'Unknown'

    if not long_df.empty:
        long_df['Date'] = date
        long_df['Location'] = location.capitalize() 
        long_df['Sample'] = sample
        long_df['File'] = str(p) 

    return long_df


manifest = []
# 250512
manifest += [
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250512/250512_dibbot_Dendra+mCherry_curve.xlsx', None),
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250512/250512_dibbot_FL1_curve.xlsx', None),
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250512/250512_dibbot_UT_curve.xlsx', None),
]
# 250514
manifest += [
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/FCCS/250514_dibbot_dendra+mCherry_droplet_curve.xlsx', None),
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/FCCS/250514_dibbot_FL1_CellLysate_droplet_curve.xlsx', None),
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/FCCS/250514_well_dendra+mCherry_curve.xlsx', None),
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/FCCS/250514_well_FL1_CellLysate_droplet_curve.xlsx', None),
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/FCCS/250514_well_UT_curve.xlsx', None),
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/FCCS/250514_dibbot_-veControl_PeglipidOil.xlsx', 'UT/Control'),
]
# 250527
manifest += [
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/FCCS/250527_dibbot_dendra+mcherry_curve.xlsx', None),
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/FCCS/250527_dibbot_FL1_curve.xlsx', None),
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/FCCS/250527_dibbot_ut_curve.xlsx', None),
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/FCCS/250527_well_dendra+mcherry_curve.xlsx', None),
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/FCCS/250527_well_FL1_curve.xlsx', None),
    ('C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/FCCS/250527_well_ut_curve.xlsx', None),
]


all_long_list = [_read_one_fcs(fp, lbl) for fp, lbl in manifest]
all_long = pd.concat([df for df in all_long_list if df is not None and not df.empty], ignore_index=True)

avg_simple = (all_long
              .groupby(['Sample', 'Location', 'Channel', 'Time [ms]'], as_index=False)
              .agg(mean=('Value', 'mean'), sd=('Value', 'std')))

per_file = (all_long
            .groupby(['File', 'Sample', 'Location', 'Channel', 'Time [ms]'], as_index=False)
            .agg(file_mean=('Value', 'mean')))
avg_biorep = (per_file
              .groupby(['Sample', 'Location', 'Channel', 'Time [ms]'], as_index=False)
              .agg(mean=('file_mean', 'mean'),
                   sd=('file_mean', 'std'),
                   n_bioreps=('file_mean', 'count')))


def _purple_palette(series):
    order = ['Fit/Green channel 1', 'Fit/Red channel 2', 'Fit channel 1>2']
    present = [c for c in order if c in set(series.unique())]
    cmap = plt.get_cmap("Purples")
    colors = cmap(np.linspace(0.45, 0.95, len(present))) if present else []
    return dict(zip(present, colors)), present

def _apply_manual_legend(fig, palette, order, title="Channel", loc="center right"):
    handles = [Line2D([0], [0], color=palette[label], lw=3, label=label) for label in order]
    
    fig.subplots_adjust(right=0.82)
    fig.legend(handles=handles, title=title, loc=loc, bbox_to_anchor=(0.92, 0.5))
    return handles


def plot_by_location(sample='Dendra+mCherry', use='biorep'):
    data = avg_biorep if use == 'biorep' else avg_simple
    sub = data[data['Sample'].str.lower() == sample.lower()].copy()
    if sub.empty:
        print(f"No data found for sample '{sample}'")
        return

    pal, order = _purple_palette(sub["Channel"])

    g = sns.FacetGrid(
        sub,
        col="Location",         
        hue="Channel",          
        palette=pal,
        sharey=False,
        height=4,
        aspect=1.2,
    )

    def _draw(data, color=None, **kws):
        plt.plot(data['Time [ms]'], data['mean'], color=color)
        if 'sd' in data and data['sd'].notna().any():
            lo = data['mean'] - data['sd']
            hi = data['mean'] + data['sd']
            plt.fill_between(data['Time [ms]'], lo, hi, alpha=0.2, color=color)
        plt.xscale('log')
        plt.xlabel('Time [ms]')
        plt.ylabel('Value')

    g.map_dataframe(_draw)

   
    _apply_manual_legend(g.fig, pal, order, title="Channel")
    if g._legend is not None:
        g._legend.remove()

    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(f"{sample}: {'Bio-rep mean±SD' if use=='biorep' else 'Simple mean±SD'}")
    plt.show()

def plot_panel_all(use='biorep', samples_order=None):
    data = avg_biorep if use == 'biorep' else avg_simple

    detected = list(data['Sample'].dropna().unique())
    default_order = [s for s in ["Dendra+mCherry", "FL1", "UT/Control"] if s in detected]
    for s in detected:
        if s not in default_order:
            default_order.append(s)
    samples_order = samples_order or default_order

    sub = data[data['Sample'].isin(samples_order)].copy()
    if sub.empty:
        print("No data to plot. Check your manifest and parsing.")
        return

    pal, order = _purple_palette(sub["Channel"])
    sub["FacetKey"] = sub["Sample"] + " • " + sub["Location"]

    g = sns.FacetGrid(
        sub,
        col="FacetKey",
        col_wrap=2,      
        hue="Channel",
        palette=pal,
        sharex=True,
        sharey=False,
        height=4,
        aspect=1.2,
    )

    def _draw(data, color=None, **kws):
        plt.plot(data["Time [ms]"], data["mean"], color=color)
        if "sd" in data and data["sd"].notna().any():
            lo = data["mean"] - data["sd"]
            hi = data["mean"] + data["sd"]
            plt.fill_between(data["Time [ms]"], lo, hi, alpha=0.2, color=color)
        plt.xscale("log")
        plt.xlabel("Time [ms]")
        plt.ylabel("Value")

    g.map_dataframe(_draw)

   
    _apply_manual_legend(g.fig, pal, order, title="Channel")
    if g._legend is not None:
        g._legend.remove()

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Combined datasets ({'Bio-rep mean±SD' if use=='biorep' else 'Simple mean±SD'})")

    g.fig.savefig("combined_plot.svg", format="svg", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_panel_all(use='biorep')




  
def export_tidy_excels():
    # where to save
    output_dir = r"C:\Users\trowy\OneDrive\Desktop\2024 and 2025 PhD\Python\Data combined\FCCS"
    os.makedirs(output_dir, exist_ok=True)

    # export each tidy file from manifest
    for fp, lbl in manifest:
        tidy = _read_one_fcs(fp, lbl)
        if tidy is None or tidy.empty:
            continue
        base = PureWindowsPath(fp).stem
        out_path = os.path.join(output_dir, f"{base}_tidy.xlsx")
        tidy.to_excel(out_path, index=False)

    # export combined data tables
    all_long.to_excel(os.path.join(output_dir, "all_long_combined.xlsx"), index=False)
    per_file.to_excel(os.path.join(output_dir, "per_file_means.xlsx"), index=False)
    avg_biorep.to_excel(os.path.join(output_dir, "avg_biorep_means.xlsx"), index=False)

    print(f"✅ Saved cleaned files to:\n{output_dir}")

if __name__ == "__main__":
    export_tidy_excels()

############################################Aleksa start here###########################################
#Plotting just from saved excels
#Is for university laptop
#%%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sns.set_theme(style="white", context="notebook")

# Folder path
DATA_DIR = r"C:\Users\jwt149\Desktop\FCCS\FCCS"

# Load file
data = pd.read_excel(os.path.join(DATA_DIR, "avg_biorep_means.xlsx"))

def _purple_palette(series):
    order = ['Fit/Green channel 1', 'Fit/Red channel 2', 'Fit channel 1>2']
    present = [c for c in order if c in set(series.unique())]
    cmap = plt.get_cmap("Purples")
    colors = cmap(np.linspace(0.45, 0.95, len(present)))
    return dict(zip(present, colors)), present

# Manual colour mapping
pal = {
    'Fit/Green channel 1': '#9B59B6',   # Purple
    'Fit/Red channel 2': '#E7549E',     # Pink
    'Fit channel 1>2': '#00A6D6'        # Blue (cross-correlation)
}

# Keep only channels that exist in your data
order = [c for c in pal.keys() if c in data["Channel"].unique()]
data["FacetKey"] = data["Sample"] + " • " + data["Location"]

g = sns.FacetGrid(
    data,
    col="FacetKey",
    col_wrap=2,
    hue="Channel",
    palette=pal,
    sharex=True,
    sharey=False,
    height=4,
    aspect=1.2,
)

def _draw(data, color=None, **kws):
    plt.plot(data["Time [ms]"], data["mean"], color=color)
    if data["sd"].notna().any():
        lo = data["mean"] - data["sd"]
        hi = data["mean"] + data["sd"]
        plt.fill_between(data["Time [ms]"], lo, hi, alpha=0.2, color=color)
    plt.xscale("log")
    plt.xlabel("Time [ms]")
    plt.ylabel("Value")

g.map_dataframe(_draw)

handles = [Line2D([0], [0], color=pal[label], lw=3, label=label) for label in order]
g.fig.subplots_adjust(right=0.82)
g.fig.legend(handles=handles, title="Channel", loc="center right", bbox_to_anchor=(0.92, 0.5))

g.fig.suptitle("Combined datasets (Bio-rep mean±SD)")
g.fig.savefig(os.path.join(DATA_DIR, "combined_plot.svg"), format="svg", dpi=300)

plt.show()


output_path = r"C:\Users\jwt149\Desktop\FCCS\My_Final_Figure.svg"

g.fig.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")


















# %%
