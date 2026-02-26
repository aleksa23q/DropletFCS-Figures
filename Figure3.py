import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import math
plt.ion()
fcs_from_250512_dibbot = [
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250512/250512_dibbot_aGFP_curve.xlsx", "250512_aGFP_dibbot"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250512/250512_dibbot_Dendra_curve.xlsx", "250512_Dendra_dibbot"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250512/250512_dibbot_mCherry_curve.xlsx","250512_mCherry_dibbot"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250512/250512_dibbot_SRB_curve.xlsx","250512_SRB_dibbot"),
]
fcs_from_250514_dibbot = [
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/250514_dibbot_dendra_droplet_curve.xlsx", "250514_Dendra dibbot"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/250514_dibbot_GFP_CellLysate_droplet_curve.xlsx", "250514_aGFP dibbot"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/250514_dibbot_mCherry_droplet_curve.xlsx","250514_mCherry dibbot"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/250514_dibbot_SRB_droplet_curve.xlsx","250514_SRB dibbot"),
]
fcs_from_250527_dibbot = [
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_dibbot_aGFP_curve.xlsx", "250527_aGFP dibbot"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_dibbot_dendra_curve.xlsx", "250527_Dendra dibbot"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_dibbot_mcherry_curve.xlsx","250527_mCherry dibbot"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_dibbot_mcherrylysate_curve.xlsx","250527_mCherry_Lysate dibbot"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_dibbot_SRB_curve.xlsx","250527_SRB dibbot"),
]
fcs_from_250527_well = [
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_well_aGFP_curve.xlsx", "250527_aGFP_well"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_well_dendra_curve.xlsx", "250527_Dendra_well"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_well_mcherry_curve.xlsx","250527_mCherry_well"),
    ("C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_well_mcherrylysate_curve.xlsx","250527_mCherry_Lysate_well"),
]

# ============================================================================
# FIGURE SETTINGS
# ============================================================================
font = {'family': 'arial', 'weight': 'normal', 'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm = 1/2.54

def process_fcs(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, skiprows=1)
    fit_cols = [c for c in df.columns if ('Fit Channel' in c and 'Residuals' not in c)]
    if 'Time [ms]' not in df.columns or not fit_cols:
        raise ValueError(f"Expected 'Time [ms]' and at least one 'Fit Channel' in {file_path}")
    clean = df[['Time [ms]'] + fit_cols].copy()
    clean.columns = ['Time ms'] + [f"r{i+1}" for i in range(len(fit_cols))]
    g = pd.melt(clean, id_vars='Time ms', var_name='sample', value_name='value').dropna(subset=['value'])
    g = g[g['Time ms'] > 0].copy()
    g['logtime'] = np.log(g['Time ms'])
    g['value_norm'] = g.groupby('sample')['value'].transform(lambda x: x / x.max())
    return g

def plot_fcs_merged(ax, file_paths, panel_title):
    frames = [process_fcs(fp) for fp in file_paths]
    g = pd.concat(frames, ignore_index=True)
    m = g.groupby('logtime', as_index=False)['value_norm'].mean()
    s = g.groupby('logtime', as_index=False)['value_norm'].std()
    ax.plot(m['logtime'], m['value_norm'], linewidth=2, label=panel_title)
    ax.fill_between(m['logtime'], m['value_norm'] - s['value_norm'],
                    m['value_norm'] + s['value_norm'], alpha=0.15)
    ax.set_title(panel_title)
    ax.set_xlabel("log(Time ms)")
    ax.set_ylabel("Normalized Value")
    ax.legend(fontsize=7)

def _key(title: str) -> str:
    t = title.strip()
    if len(t) >= 7 and t[:6].isdigit() and t[6] == "_":
        t = t.split("_", 1)[1]
    t = t.lower().replace("_", " ")  
    aliases = {
        "mcherry lystae dibbot": "mcherry lysate dibbot",
        "mcherry lystae well":  "mcherry lysate well",
    }
    return aliases.get(t, t)
map512_dib = {_key(title): (fp, title) for fp, title in fcs_from_250512_dibbot}
map514_dib = {_key(title): (fp, title) for fp, title in fcs_from_250514_dibbot}
map527_dib = {_key(title): (fp, title) for fp, title in fcs_from_250527_dibbot}
map512_well = {} 
map514_well = {}  
map527_well = {_key(title): (fp, title) for fp, title in fcs_from_250527_well}


all_dib_keys = set(map512_dib) | set(map514_dib) | set(map527_dib)
counts = {k: int(k in map512_dib) + int(k in map514_dib) + int(k in map527_dib) for k in all_dib_keys}
merged_dib = sorted([k for k, c in counts.items() if c >= 2])  
single_dib = sorted([k for k, c in counts.items() if c == 1])  


desired_order = [
    ("dibbot", "agfp dibbot"),
    ("well",   "agfp well"),
    ("dibbot", "dendra dibbot"),
    ("well",   "dendra well"),
    ("dibbot", "mcherry dibbot"),
    ("well",   "mcherry well"),
    ("dibbot", "mcherry lysate dibbot"), 
    ("well",   "mcherry lysate well"),
    ("dibbot", "srb dibbot"),
]
available = set(panels)
panels = [p for p in desired_order if p in available] + [p for p in panels if p not in set(desired_order)]

n = len(panels)
ncols = 2
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharey=True)
axes = np.array(axes).reshape(-1)

for (kind, key), ax in zip(panels, axes):
    fps = []
    if kind == "dibbot":
        if key in map512_dib: fps.append(map512_dib[key][0])  # <-- add this
        if key in map514_dib: fps.append(map514_dib[key][0])
        if key in map527_dib: fps.append(map527_dib[key][0])
    else:
        if key in map514_well: fps.append(map514_well[key][0])
        if key in map527_well: fps.append(map527_well[key][0])
    ...

        

    title_clean = f"{key.title()} ({kind})"
    if len(fps) == 1:
        title_clean += " (single repeat)"
    plot_fcs_merged(ax, fps, panel_title=title_clean)

for ax in axes[n:]:
    ax.set_visible(False)

for ax in axes[:n]:
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()

from pathlib import Path
out_path = Path(r"C:\Users\trowy\OneDrive\Desktop\2024 and 2025 PhD\Python\Exports") / "fcs_panels.svg"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, format="svg", bbox_inches="tight")

export_dir = r"C:\Users\trowy\OneDrive\Desktop\2024 and 2025 PhD\Python\Data combined\FCS"
os.makedirs(export_dir, exist_ok=True)

def process_fcs(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, skiprows=1)

    fit_cols = [c for c in df.columns if ('Fit Channel' in c and 'Residuals' not in c)]
    if not fit_cols:
        raise ValueError(f"No 'Fit Channel' columns found in {file_path}")
    if 'Time [ms]' not in df.columns:
        raise ValueError(f"'Time [ms]' column not found in {file_path}")
    
    clean = df[['Time [ms]'] + fit_cols].copy()
    clean = clean[clean['Time [ms]'] > 0].copy()
    
    clean.columns = ['Time ms'] + [f"r{i+1}" for i in range(len(fit_cols))]
    
    g = pd.melt(clean, id_vars='Time ms', var_name='replicate', value_name='value').dropna(subset=['value'])
    
    g['logtime'] = np.log(g['Time ms'])
    
    g['value_norm'] = g.groupby('replicate')['value'].transform(lambda x: x / x.max())

    return g

dibbot_files_250512 = {
    "Dendra": [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250512/250512_dibbot_Dendra_curve.xlsx"],
    "aGFP":   [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250512/250512_dibbot_aGFP_curve.xlsx"],
    "mCherry":[r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250512/250512_dibbot_mCherry_curve.xlsx"],
    "SRB":[r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250512/250512_dibbot_SRB_curve.xlsx"],
}
dibbot_files_250514 = {
    "Dendra": [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/250514_dibbot_dendra_droplet_curve.xlsx"],
    "aGFP":   [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/250514_dibbot_GFP_CellLysate_droplet_curve.xlsx"],
    "mCherry":[r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/250514_dibbot_mCherry_droplet_curve.xlsx"],
    "SRB":    [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250514/250514_dibbot_SRB_droplet_curve.xlsx"],
}
dibbot_files_250527 = {
    "aGFP":           [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_dibbot_aGFP_curve.xlsx"],
    "Dendra":         [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_dibbot_dendra_curve.xlsx"],
    "mCherry":        [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_dibbot_mcherry_curve.xlsx"],
    "mCherry lysate": [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_dibbot_mcherrylysate_curve.xlsx"],
    "SRB":            [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_dibbot_SRB_curve.xlsx"],
}

all_samples = sorted(set(dibbot_files_250514) | set(dibbot_files_250527) | set(dibbot_files_250512))
dibbot_combined = {
    s: dibbot_files_250512.get(s, []) + dibbot_files_250514.get(s, []) + dibbot_files_250527.get(s,)
    for s in all_samples
}

well_files = {
    "aGFP":           [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_well_aGFP_curve.xlsx"],
    "Dendra":         [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_well_dendra_curve.xlsx"],
    "mCherry":        [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_well_mcherry_curve.xlsx"],
    "mCherry lysate": [r"C:/Users/trowy/OneDrive/Desktop/2024 and 2025 PhD/Python/250527/250527_well_mcherrylysate_curve.xlsx"],
}

def export_group(sample_dict: dict, label: str):
    for sample, files in sample_dict.items():
        if not files:
            continue
        dfs = []
        for f in files:
            g = process_fcs(f)
            g['source_file'] = f
            g['sample'] = sample
            g['group'] = label
            dfs.append(g)
        combined = pd.concat(dfs, ignore_index=True)
        safe_sample = sample.replace(" ", "_")
        out_path = os.path.join(export_dir, f"{safe_sample}_{label}_combined.csv")
        combined.to_csv(out_path, index=False)
        print(f"Saved: {out_path}  ({len(combined):,} rows)")

export_group(dibbot_combined, label="dibbot")
export_group(well_files,     label="well")