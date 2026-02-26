# Scatter Plot: Droplet eccentricity vs area by method
# Compares DIB-BOT vs hand-pipetted droplets

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

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
input_folder = f'{root_path}Results/'
output_folder = f'{root_path}Figures/'

os.makedirs(output_folder, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
# Load droplet characterisation output
char_path = f"{input_folder}/droplet_char_488.csv"
char_df = pd.read_csv(char_path)

print("Columns available:", char_df.columns.tolist())
print(f"Total rows: {len(char_df)}")

# Parse method from stack name (same convention as area comparison)
# Filename format: date_method_oilcomp_protein_replicate
char_df[['date', 'method', 'oil_comp', 'protein', 'replicate']] = char_df['stack'].str.split('_', expand=True)

print("\nUnique methods:", char_df['method'].unique())
print("Data count by method:")
print(char_df['method'].value_counts())

# ============================================================================
# FILTER DATA
# ============================================================================
methods = ['dibbot', 'HP']
df_plot = char_df[char_df['method'].str.lower().isin([m.lower() for m in methods])].copy()
print(f"\nFiltered to {len(df_plot)} rows")

# Keep largest droplet per image (consistent with area comparison)
print(f"Before filtering: {len(df_plot)} droplets")
df_plot = df_plot.loc[df_plot.groupby(['stack', 'frame'])['area'].idxmax()]
print(f"After keeping largest per image: {len(df_plot)} droplets")

# ============================================================================
# LABEL MAPPING
# ============================================================================
# Map method codes to display labels
label_map = {}
for m in df_plot['method'].unique():
    if m.lower() == 'dibbot':
        label_map[m] = 'DIB-BOT'
    elif m.lower() == 'hp':
        label_map[m] = 'Hand Pipette'
    else:
        label_map[m] = m

df_plot['method_label'] = df_plot['method'].map(label_map)

# ============================================================================
# SCATTER PLOT
# ============================================================================
fig, ax = plt.subplots(figsize=(8*cm, 6*cm))

colors = {'DIB-BOT': '#1f77b4', 'Hand Pipette': '#ff7f0e'}

for method_label in df_plot['method_label'].unique():
    subset = df_plot[df_plot['method_label'] == method_label]
    color = colors.get(method_label, 'gray')
    ax.scatter(
        subset['area'],
        subset['eccentricity'],
        label=method_label,
        color=color,
        alpha=0.5,
        s=15,
        edgecolors='none'
    )
    print(f"\n{method_label}: {len(subset)} droplets")
    print(f"  Area: {subset['area'].mean():.1f} ± {subset['area'].std():.1f}")
    print(f"  Eccentricity: {subset['eccentricity'].mean():.3f} ± {subset['eccentricity'].std():.3f}")

ax.set_xlabel('Droplet Area (px)')
ax.set_ylabel('Eccentricity')
ax.set_title('Droplet Eccentricity vs Area')
ax.legend(frameon=False, fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{output_folder}/scatter_eccentricity_area.svg")
plt.savefig(f"{output_folder}/scatter_eccentricity_area.png", dpi=300)
plt.show()

print(f"\nFigure saved to {output_folder}")