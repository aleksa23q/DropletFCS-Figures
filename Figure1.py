#Figure1: Average boundary intensities
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statannotations.Annotator import Annotator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import scipy.stats as stats
import matplotlib.transforms as transforms

#root path
root_path = open('experimental_data/experiment_index.txt', 'r').readlines()[0].strip()

input_folder = f'{root_path}Results/'
output_folder = f'{root_path}Figures/'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

#Load the boundary_pixels data frame
boundary_pixels_path = f"{input_folder}/boundary_pixels_488.csv"
boundary_pixels = pd.read_csv(boundary_pixels_path)

#Sort dataframe
boundary_pixels[['date', 'method', 'oil_comp', 'protein', 'replicate']] = boundary_pixels["stack"].str.split('_', expand=True)
boundary_pixels["condition"] = [f'{method}_{oil_comp}_{protein}' for method, oil_comp, protein in boundary_pixels[['method', 'oil_comp', 'protein']].values]

#Calculate average intensities for each condition and replicate
boundary_pixels = boundary_pixels[['condition', 'replicate', 'boundary_mean_intensity']]
boundary_pixels = boundary_pixels.groupby(["condition", "replicate"]).mean().reset_index()
boundary_pixels.rename(columns={'boundary_mean_intensity': 'boundary_intensity'}, inplace=True)

#Bardotplot function
def bardotplot(data, xcol, ycol, order, hue=None, hue_order=None, scat_hue=None, scat_hue_order=None, palette=False, xlabel='', ylabel=False, pairs=False, correction=None, xticks=None, groups=None, group_label_y=-0.18, group_line_y=-0.05, ax=None, legend='', dot_size=5, cap_size=0.2, cap_width=2):
    if ax is None:
        fig, ax = plt.subplots()
    sns.barplot(
        data=data,
        x=xcol,
        y=ycol,
        hue=hue,
        palette=palette,
        capsize=cap_size,
        errwidth=cap_width,
        ax=ax,
        dodge=True,
        order=order,
        hue_order=hue_order,
        edgecolor='white'
    )
    sns.stripplot(
        data=data,
        x=xcol,
        y=ycol,
        hue=scat_hue,
        palette=palette,
        ax=ax,
        edgecolor='#fff',
        linewidth=1,
        s=dot_size,
        order=order,
        hue_order=scat_hue_order,
        dodge=True,
    )

    if pairs:
        annotator = Annotator(
            ax=ax, pairs=pairs, data=data, x=xcol, y=ycol, order=order, hue=hue, hue_order=hue_order)
        annotator.configure(test='t-test_ind', text_format='star',
                    loc='inside', comparisons_correction=correction, line_width=0.5,
                    pvalue_thresholds=[[1e-4, '****'], [1e-3, '***'], [1e-2, '**'], [0.05, '*'], [1, 'ns']])
        annotator.apply_and_annotate()

    ax.set(ylabel=ylabel)
    ax.set_xlabel(xlabel)
    if xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels(hue_order*len(order))
    if groups:
        for group_label, (x0, x1, x2) in groups.items():
            ax.annotate(group_label, xy=(x0, group_label_y),
                        xycoords='data', ha='center', annotation_clip=False)
            trans = ax.get_xaxis_transform()
            ax.plot([x1, x2], [group_line_y, group_line_y],
                    color="black", transform=trans, clip_on=False)

    if legend == '':
        ax.legend('', frameon=False)
    else:    
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    return ax

boundary_pixels[['method', 'oil', 'protein']] = boundary_pixels['condition'].str.split('_', expand=True)

font = {'family' : 'arial',
'weight' : 'normal',
'size'   : 8 }
plt.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm = 1/2.54

# Extract oil component and normalize
boundary_pixels['oil_order'] = boundary_pixels['condition'].str.split('_').str[1].str.lower()
boundary_pixels['protein'] = boundary_pixels['protein'].str.lower()
boundary_pixels['oil'] = boundary_pixels['oil'].str.lower()

# Define hue order: oil -> ol (oil + lipid) -> peglipid
hue_order = ['oil', 'ol', 'peglipid']

pairs = [
    (('dendra', 'oil'), ('dendra', 'ol')),
    (('dendra', 'oil'), ('dendra', 'peglipid')),
    (('dendra', 'ol'), ('dendra', 'peglipid')),
    (('dendra+mcherry', 'oil'), ('dendra+mcherry', 'ol')),
    (('dendra+mcherry', 'oil'), ('dendra+mcherry', 'peglipid')),
    (('dendra+mcherry', 'ol'), ('dendra+mcherry', 'peglipid'))
]

fig, ax = plt.subplots(figsize=(18*cm, 6*cm))
bardotplot(
    data=boundary_pixels,
    xcol='protein',
    ycol='boundary_intensity',
    order=['dendra', 'dendra+mcherry'],
    hue='oil_order',
    hue_order=hue_order,
    scat_hue='oil_order',
    scat_hue_order=hue_order,
    palette='Set2',
    xlabel='Protein',
    ylabel='Mean Boundary Intensity',
    pairs=pairs,
    correction='holm-bonferroni',
    ax=ax
)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(handles=by_label.values(), labels=by_label.keys(), title='Oil Type', loc='upper right', bbox_to_anchor=(1.2, 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

output_path = os.path.join(output_folder, 'average_boundary_intensities.svg')
plt.savefig(output_path, bbox_inches='tight', dpi=300, format='svg')
print(f"Saved to {output_path}")