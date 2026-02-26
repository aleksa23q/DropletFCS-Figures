##################################All for Aleksa#######################
##################################
##################################
##################################
#This makes FCCS curves of 5 to 35 mins
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# ============================================================================
# FIGURE SETTINGS
# ============================================================================
font = {'family': 'arial', 'weight': 'normal', 'size': 8}
matplotlib.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm = 1/2.54

DATA_DIR = r"C:/Users/jwt149/Desktop/Tev protease excel files cleaned up from python"
sns.set_theme(style="white", context="notebook")

# List all Excel files
cleaned_files = [os.path.join(DATA_DIR, f) 
                 for f in os.listdir(DATA_DIR) 
                 if f.endswith(".xlsx")]

if not cleaned_files:
    print("No Excel files found in the folder!")
else:
    ncols = 2
    n = min(len(cleaned_files), 7)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
    axes = axes.flatten() if n > 1 else [axes]

    # Define custom colors for each channel
    channel_colors = {
        'Fit Channel 1': '#9B59B6',       # Purple
        'Fit Channel 2': '#E7549E',       # Pink
        'Fit Channel 1 -> 2': '#00A6D6'   # Blue
    }

    for fp, ax in zip(cleaned_files[:7], axes):
        df = pd.read_excel(fp)
        sns.lineplot(
            data=df,
            x='Time [ms]',
            y='Value',
            hue='Channel',
            ci='sd',
            palette=channel_colors,
            ax=ax
        )
        ax.set_xscale('log')  # log x-axis
        ax.set_title(os.path.basename(fp))
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('G(t)')
        leg = ax.get_legend()
        if leg:
            leg.remove()
        ax.grid(False)

    # Remove extra axes if fewer than grid
    for ax in axes[n:]:
        fig.delaxes(ax)

    plt.tight_layout()

    svg_path = os.path.join(OUTPUT_DIR, "panel_plot.svg")
    fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
    print(f"Saved panel plot → {svg_path}")

    plt.show()
##################################
##################################
##################################

##################################
##################################
##################################
#Cross correlation FCS curves from 5 min to 35 mins
import os, glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LogLocator, LogFormatter
from matplotlib.colors import Normalize

DATA_DIR = r"C:/Users/jwt149/Desktop/Tev protease excel files cleaned up from python"
OUTPUT_DIR = r"C:/Users/jwt149/Desktop/Tev protease excel files cleaned up from python/Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# get all Excel files
excel_files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))

# extract times from filenames and sort by time
file_times = []
for fp in excel_files:
    match = re.search(r'(\d+)\s*min', os.path.basename(fp))
    if match:
        file_times.append((int(match.group(1)), fp))
file_times.sort(key=lambda x: x[0])

# reversed colormap
cmap = cm.get_cmap('Blues_r')

# define the gradient mapping: 5 min = darkest, 35 min = color at 25 min
times = [t for t,_ in file_times]
vmin, vmax = min(times), max(times)

# find normalized value corresponding to 25 min
norm_25 = (25 - vmin) / (vmax - vmin)

class CustomNormalize(Normalize):
    """Map vmin->vmax to 0->norm_25 in colormap"""
    def __init__(self, vmin, vmax, vmax_mapped):
        super().__init__(vmin=vmin, vmax=vmax)
        self.vmax_mapped = vmax_mapped  # max colormap fraction for 35 min

    def __call__(self, value, clip=None):
        normed = (value - self.vmin) / (self.vmax - self.vmin)  # 0->1
        normed = normed * self.vmax_mapped  # map to 0 -> vmax_mapped
        return normed

norm = CustomNormalize(vmin=vmin, vmax=vmax, vmax_mapped=norm_25)

fig, ax = plt.subplots(figsize=(8,5))

for t, fp in file_times:
    df = pd.read_excel(fp)
    df_ch = df[df['Channel'].str.startswith('Fit Channel 1 -> 2')]
    df_pivot = df_ch.pivot(index='Time [ms]', columns='Replicate', values='Value')
    mean_vals = df_pivot.mean(axis=1)
    std_vals = df_pivot.std(axis=1)
    
    color = cmap(norm(t))
    ax.plot(df_pivot.index, mean_vals, color=color, lw=2, solid_capstyle='round', label=f"{t} min")
    ax.fill_between(df_pivot.index, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.3)

# log-x axis
ax.set_xscale('log')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('G(t)')
ax.set_title('Channel 1 -> 2: mean ± SD per file')
ax.tick_params(axis='both', which='both', direction='out', length=6, width=1.5)
ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2,10), numticks=100))
ax.xaxis.set_major_formatter(LogFormatter())
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(title="Time", fontsize=8)

# save as SVG
out_fp = os.path.join(OUTPUT_DIR, "overlay_channel1to2_meanSD_timeGradient_custom.svg")
fig.savefig(out_fp, format='svg', dpi=300, bbox_inches='tight')
print(f"Saved overlay plot → {out_fp}")

plt.show()
##################################
##################################
##################################


##################################
##################################
##################################
#Need to run line graph before this
#Dot points of each tev protease time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

last_points = {5: 0.005, 10: 0.0035, 15: 0.0027, 20: 0.0023, 25: 0.0020, 30: 0.0015, 35: 0.001}

desired_mins = list(last_points.keys())

# Blue gradient: 5 min darkest, 35 min lightest
cmap = cm.get_cmap('Blues')
norm = Normalize(vmin=min(desired_mins), vmax=max(desired_mins))
colors = {m: cmap(1 - norm(m) * 0.7) for m in desired_mins}  # compress gradient

fig, ax = plt.subplots(figsize=(7.5,5))
for m in desired_mins:
    ax.scatter(m, last_points[m], color=colors[m], s=80, label=f"{m} min")

ax.set_xlabel("Time (min)")
ax.set_ylabel("G(t) at end of curve")
ax.set_title("End-point values from line graph")
ax.set_xticks(desired_mins)
ax.set_xlim(0, 40)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.tick_params(axis='both', which='both', direction='out', length=6, width=1.2)
ax.legend(title="Time", frameon=False)
ax.grid(False)

out_fp = os.path.join(OUTPUT_DIR, "overlay_channel1to2_meanSD_timeGradient_custom.svg")
fig.savefig(out_fp, format='svg', dpi=300, bbox_inches='tight')
print(f"Saved overlay plot → {out_fp}")

plt.show()



























# %%
