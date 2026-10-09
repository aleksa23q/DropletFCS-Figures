# %%
import os, glob, re #import os — lets you interact with the file system: listing directories, joining paths, checking if files exist.
#import glob — finds files using wildcard patterns, e.g. *.xlsx to get all Excel files in a folder.
#import re — regular expressions, used here to extract numbers from filenames like pulling 35 out of _35min_.
import pandas as pd # the main data manipulation library. Loads Excel files into DataFrames, filters rows, pivots tables etc.
import matplotlib #the base matplotlib library, used here just to update global font settings via matplotlib.rcParams.
import matplotlib.pyplot as plt #the main plotting interface: creates figures, axes, shows/saves plots.
import matplotlib.gridspec as gridspec #lets you create complex figure layouts with panels of different sizes, which is how panels A-E are arranged.
import seaborn as sns #built on top of matplotlib, used here for sns.lineplot in Panel B which handles the mean ± SE shading automatically.
from matplotlib import cm #colour maps (e.g. Blues, Blues_r), used to colour the timepoint curves in Panels C and D.
from matplotlib.colors import Normalize # maps a range of values (e.g. 0–35 min) onto a 0–1 scale so they can be fed into a colormap.
from matplotlib.ticker import LogLocator, NullLocator #controls axis tick marks: LogLocator places ticks at log intervals, NullLocator removes minor ticks entirely.
import numpy as np #numerical computing: arrays, interpolation (np.interp), mean, std, concatenation etc. Used everywhere.
from collections import defaultdict # a dictionary that automatically creates a default value (here an empty list) when you access a key that doesn't exist yet, used to group files by condition and timepoint.
from scipy.optimize import curve_fit #fits a mathematical function to data points, used in Panel D to fit the exponential decay to TEV data and the flat line to no TEV data.

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
    'legend.fontsize': 10
}
plt.rcParams.update(font_settings)
matplotlib.rcParams.update(font_settings)
sns.set_theme(style="ticks", context="notebook", rc=font_settings)
cent = 1 / 2.54

# ============================================================================
# PATHS
# ============================================================================
DIBBOT_DIR = r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\FCCS\3.Dibbot_well_combined\To upload to zenodo"
TEV_DIR    = r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\Tev_protease\Tev_protease_cleaned_excel_repeats\zenodo"
OUTPUT_DIR = r"C:\Users\jwt149\Desktop\FCS_Cleaned_data_together\SVG_for_final_figures\Figures used for final submission"

# ============================================================================
# COLORS
# ============================================================================
channel_colors = {
    'Fit Channel 1':      '#9B59B6',
    'Fit Channel 2':      '#E7549E',
    'Fit Channel 1 -> 2': '#00A6D6'
}

# ============================================================================
# HELPER
# ============================================================================
def process_files(file_list):
    dfs = [pd.read_excel(fp) for fp in file_list] #Reads every Excel file in the list into a pandas DataFrame and stores them all in a list.
    all_times   = np.unique(np.concatenate([df['Time [ms]'].values for df in dfs])) #Pulls the time column from every file, concatenates them into one big array, then keeps only unique values. This gives every time point that exists across all files combined.
    common_time = np.sort(all_times) #Sorts those unique time points in ascending order. This becomes the shared time axis all replicates will be interpolated onto
    aligned_dfs = [] #Empty list that will collect the processed data from each replicate.
    for df in dfs:
        replicates = df['Replicate'].unique() if 'Replicate' in df.columns else [0] #Gets the list of replicate IDs from the file. If there's no Replicate column it just assigns a dummy replicate [0].
        for rep in replicates: # loops through replicates in the current file. Each replicate will be processed separately to ensure we keep them distinct when we later compute means and SEs.
            df_rep     = df[df['Replicate'] == rep] if 'Replicate' in df.columns else df #Filters down to just rows I need
            df_aligned = pd.DataFrame({'Time [ms]': common_time}) #Creates a fresh DataFrame with the common time axis as its only column. as each excel file has multiple time columns
            for ch in df['Channel'].unique():
                vals = df_rep[df_rep['Channel'] == ch].set_index('Time [ms]')['Value'] # this line and above line looks through the Fit Channel 1, Fit Channel 2, Fit Channel 1 -> 2 and matches to the time. It then sets the time column as the index and keeps only the Value column. So we end up with a Series where the index is time and the value is G(t) for that channel and replicate.
                if vals.empty:
                    continue #skips channel if theres no data for it
                df_aligned[ch] = np.interp(common_time, vals.index, vals.values) #Interpolates the channel's values onto the common time axis. So if this replicate was measured at slightly different time points than another file, it gets mapped to the same grid. This is important as offset time points between replicates was giving weird zig zags in fit lines.
            df_long = df_aligned.melt( 
                id_vars='Time [ms]',
                value_vars=[c for c in df['Channel'].unique() if c in df_aligned.columns],
                var_name='Channel', value_name='Value'
            ) ##Just melts the data which is what seaborn i believe needs to look at data
            aligned_dfs.append(df_long) #Adds this long-format DataFrame to the list of aligned DataFrames. Each row in df_long corresponds to a single time point, channel, and value for one replicate.
    return pd.concat(aligned_dfs, ignore_index=True) #After processing all files and replicates, concatenates all the long-format DataFrames into one big DataFrame that contains all the data aligned to the common time axis. This final DataFrame is what gets plotted in Panel B.

# ============================================================================
# LOAD DIBBOT DATA
# ============================================================================
dibbot_groups = defaultdict(list) #Creates a dictionary where each key will be a condition (e.g. 'acGFP1-mCherry') and the value will be a list of file paths that belong to that condition. The defaultdict ensures that if we access a key that doesn't exist yet, it will automatically create an empty list for it.
for f in os.listdir(DIBBOT_DIR): #loops through file in diibot directory
    if not f.endswith(".xlsx"): #Skips anything not an excel file
        continue
    name = f.lower() #converts to lowercase to not miss any files due to case differences in the name
    if 'dibbot' not in name:
        continue # skips file without dibbot in it just to be safe, as some files in that folder are not dibbot data and are well/chamber data.
    fp = os.path.join(DIBBOT_DIR, f) #Joins the directory path with the filename to get the full file path, which will be stored in the appropriate group in the dibbot_groups dictionary.
    if   'fl1'            in name: dibbot_groups['acGFP1-mCherry'].append(fp)
    elif 'dendra_mcherry' in name: dibbot_groups['Dendra + mCherry'].append(fp)
    elif 'agfp_mcherry'   in name: dibbot_groups['acGFP1 and mCherry'].append(fp)
    elif 'ut'             in name: dibbot_groups['Untransfected'].append(fp) #Just matching everything. for example my FL1 stook dor fluorescently linked as this is what Luke gave me however the term for this sample in the paper is acGFP1-mCherry. Untransfected (and dendra) not used in paper or plotted but will keep just incase.

dibbot_keys = ['acGFP1 and mCherry', 'acGFP1-mCherry'] #Defines which two Dibbot groups actually get used in Panel B — the other two (Dendra, Untransfected) are loaded but never plotted.

no_tev_files_dibbot = []
for f in os.listdir(TEV_DIR):
    if not f.endswith(".xlsx"):
        continue
    name = f.lower()
    if 'no_tev' in name or 'notev' in name:
        no_tev_files_dibbot.append(os.path.join(TEV_DIR, f)) #Same as above, directory for No_tev, loops through,  skips non-excel, different naming conventions across daya so catches notev and no_tev, adds file path as before. only thing of note is there are 0 to 35min xcel files so loads all but i only end up plotting 0 min so not sure if i need to load all here

# ============================================================================
# LOAD TEV DATA
# ============================================================================
excel_files        = glob.glob(os.path.join(TEV_DIR, "*.xlsx")) #This line uses glob to find all Excel files in the TEV_DIR directory and stores their full file paths in a list called excel_files. This is a more efficient way to get all the relevant files compared to manually looping through the directory and checking each filename, as it handles the pattern matching internally.
time_groups        = defaultdict(list) # Dictionary to store Excel files grouped by time points
time_groups_no_tev = defaultdict(list) # Separate dictionary to store "No TEV" files grouped by time points, since they are treated differently in the analysis and plotting. This allows us to easily access and plot them separately in Panel C and D.``````

for fp in excel_files: #Loops through each Excel file found in the TEV_DIR directory. For each file, it will determine which time point it belongs to by extracting the number of minutes from the filename using a regular expression. It then categorizes the file into either the time_groups dictionary (for regular TEV files) or the time_groups_no_tev dictionary (for No TEV files) based on whether "no_tev" is present in the filename. 
    name  = os.path.basename(fp).lower() #lowcase file name to ensure none are missed
    match = re.search(r'(\d+)\s*min', name) #Ensures that disprecnies in code naming arent mis _5_min and 5min will both be saved
    if match:
        t = int(match.group(1)) #retrieves the first captured group — the number — as a string. int() converts it to an integer so it can be used as a dictionary key and for sorting. e.g. '35' → 35.
        if "no_tev" in name: time_groups_no_tev[t].append(fp)
        else:                 time_groups[t].append(fp) #If the filename contains no_tev the file path gets added to the no TEV dictionary under that timepoint. Otherwise it goes into the TEV dictionary. So after this loop time_groups[0] contains all 3 TEV 0 min files, time_groups[35] all 3 TEV 35 min files, and so on.

times_sorted = sorted(set(list(time_groups.keys()) + list(time_groups_no_tev.keys()))) #This line combines the keys (time points) from both dictionaries into a single set to ensure uniqueness, then sorts them in ascending order. This gives us a sorted list of all time points for which we have data, regardless of whether it's TEV or No TEV.

cmap_blues = cm.get_cmap('Blues_r')
vmin, vmax = min(times_sorted), max(times_sorted)
norm_25    = (25 - vmin) / (vmax - vmin) #This gets blues for the tev and sets the 25 mark so that it doesnt get to blue by the 35 mark.

class CustomNormalize(Normalize): #defines new class inherited from matplotlibs normlaise that I can change
    def __init__(self, vmin, vmax, vmax_mapped): #takes min 0min and max 35min and compresses colours between 0 and 25min so that 35min doesnt get to blue or 0 mn to white
        super().__init__(vmin=vmin, vmax=vmax)
        self.vmax_mapped = vmax_mapped
    def __call__(self, value, clip=None):
        return ((value - self.vmin) / (self.vmax - self.vmin)) * self.vmax_mapped

norm_custom = CustomNormalize(vmin=vmin, vmax=vmax, vmax_mapped=norm_25) #All above code is just setting the blues so it doesn't get to blue or too white.

# ============================================================================
# FIGURE LAYOUT
# ============================================================================
fig_w = 18 * cent
fig_h = 17 * cent
fig   = plt.figure(figsize=(fig_w, fig_h)) #typical figure width for journal is 18cm and like 24cm high but not sure

row_h          = 5 * cent / fig_h
gap            = 1.8 * cent / fig_h
bottom_row_top = 1.0 - row_h
bottom_row_bot = bottom_row_top - gap - row_h

# --- TOP ROW ---
gs_top = gridspec.GridSpec(
    1, 2,
    figure=fig,
    width_ratios=[6, 14],
    left=0.0,
    right=1.0,
    bottom=1.0 - row_h,
    top=1.0,
    wspace=(0.3 * cent / fig_w) / (14 / 20)
) #Matplotlib understands spaces better in a 0-1 range instead of saying a gap needs to be 3cm, so just changing it here for easier plotting. 

ax_a = fig.add_subplot(gs_top[0, 0]) 
ax_a.set_visible(False) #Subplot for panel A as a blank space. 

ax_c_blank = fig.add_axes([ax_a.get_position().x0, 
                           bottom_row_bot + row_h, 
                           ax_a.get_position().width, 
                           row_h])
ax_c_blank.axis('off') #Places spot for panel C label to be the same distance from the plot as panel A

inner_b = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_top[0, 1], wspace=0.25) #Creates a nested grid of 1 row × 3 columns inside the top-right cell of gs_top — this is where Panel B's three subplots live. wspace=0.25 sets the horizontal spacing between the three subplots.
axes_b  = [fig.add_subplot(inner_b[0, i]) for i in range(3)] #Creates the three Panel B axes in one line, stored as a list [axes_b[0], axes_b[1], axes_b[2]] for the three conditions.

# --- BOTTOM ROW: same width as B ---
gs_bot = gridspec.GridSpec(
    1, 2, #Creating columns for panel C and D
    figure=fig,
    width_ratios=[1, 1], #width for panels are equal
    left=gs_top[0, 1].get_position(fig).x0, #below panel B not below panel A so left edge is same as panel B
    right=20/20, #Sets right edge to be same as panel B
    bottom=bottom_row_bot,
    top=bottom_row_bot + row_h,
    wspace=0.45
) #Just setting up loaction for panel C and D with width space and everything

ax_c = fig.add_subplot(gs_bot[0, 0])
ax_d = fig.add_subplot(gs_bot[0, 1]) #Location of panel C and D

# ============================================================================
# PANEL LABELS
# ============================================================================
def panel_label(fig, gs, index, letter):
    pos = gs[index].get_position(fig)
    fig.text(pos.x0, pos.y1, letter, fontsize=10, fontweight='bold', va='bottom') #Where to put a bold letter indicating where panel is and loocation which is top left of each panel

panel_label(fig, gs_top, (0, 0), 'A')
panel_label(fig, gs_top, (0, 1), 'B')
fig.text(ax_c_blank.get_position().x0, ax_c_blank.get_position().y1, 'C', 
         fontsize=10, fontweight='bold', va='bottom')
panel_label(fig, gs_bot,  (0, 0), 'D')
panel_label(fig, gs_bot,  (0, 1), 'E') #Where each bold letter goes.

# ============================================================================
# PANEL B
# ============================================================================
b_ylims  = [0.010, 0.010, 0.010]
b_yticks = [
    np.arange(0, 0.010 + 0.002, 0.002),
    np.arange(0, 0.010 + 0.002, 0.002),
    np.arange(0, 0.010 + 0.002, 0.002),
] #panel B has 3 subplots, so sets every plot to max y lim of 0.010 with ticks at each 0.002 interval.

panel_b_data  = dibbot_keys + ['acGFP1-TEV-mCherry'] #Takes keys acGFP1 and mCherry and acGFP1-mCherry and adds key for acGFP1-TEV-mCherry
panel_b_files = [dibbot_groups[k] for k in dibbot_keys] + [no_tev_files_dibbot] #Builds matching list for each subplot.

for i, (key, files, ymax, yticks) in enumerate(zip(
        panel_b_data, panel_b_files, b_ylims, b_yticks)):
    df = process_files(files)
    sns.lineplot(data=df, x='Time [ms]', y='Value', hue='Channel',
                 errorbar=('se', 1), palette=channel_colors, ax=axes_b[i])
    axes_b[i].set_xscale('log')
    axes_b[i].set_title(key)
    axes_b[i].set_xlabel('Time [ms]')
    axes_b[i].set_ylabel('G(t)')
    axes_b[i].set_ylim(0, ymax)
    axes_b[i].set_yticks(yticks)
    axes_b[i].set_xticks([1e-2, 1e0, 1e2], labels=['$10^{-2}$', '$10^{0}$', '$10^{2}$'])
    axes_b[i].xaxis.set_minor_locator(NullLocator())
    axes_b[i].tick_params(axis='x', which='major', direction='out', length=4, width=0.8, bottom=True)
    axes_b[i].tick_params(axis='y', which='major', direction='out', length=3, width=0.8, left=True)
    axes_b[i].yaxis.set_minor_locator(NullLocator())
    axes_b[i].spines['top'].set_visible(False)
    axes_b[i].spines['right'].set_visible(False)
    axes_b[i].grid(False) #All plotting prameters for panel B subplots, including log scale, axis labels, limits, ticks, and styling.

for ax in axes_b[1:]:
    ax.set_ylabel('')
    ax.yaxis.set_ticklabels([]) #Only the first subplot gets y axis labels and ticks, the other two have them removed for cleaner look.

handles, labels = axes_b[0].get_legend_handles_labels()
new_labels = [
    'acGFP1'               if l == 'Fit Channel 1'      else
    'mCherry'               if l == 'Fit Channel 2'      else
    'Cross-correlation' if l == 'Fit Channel 1 -> 2' else l
    for l in labels #For key what i want to call fit channel 1 and so on.
]
axes_b[0].legend(handles, new_labels, loc='upper right', fontsize=8,
                 frameon=False, facecolor='white', edgecolor='black')
for ax in axes_b[1:]:
    leg = ax.get_legend()
    if leg: leg.remove() #Legend/key location and removs box for it. 

# ============================================================================
# PANEL C
# ============================================================================
for t in times_sorted:
    dfs = [pd.read_excel(fp) for fp in time_groups[t]
           if "no_tev" not in os.path.basename(fp).lower()] #Reads all TEV files for that timepoint into DataFrames, with an extra filter to exclude any no_tev files that may have slipped into time_groups. This is a safety check.
    if not dfs:
        continue #If there are no files for this time point, skip to the next iteration of the loop. This prevents errors from trying to process an empty list of DataFrames.
    all_vals, col_counter = [], 0
    for df in dfs: #Loops through each DataFrame for the current time point. Each DataFrame corresponds to one Excel file (one replicate). The goal is to extract the cross-correlation data (Fit Channel 1 -> 2) and align it to a common time axis across all replicates.
        df_ch    = df[df['Channel'].str.startswith('Fit Channel 1 -> 2')] #Filters to only cross correlation channel. 
        df_pivot = df_ch.pivot(index='Time [ms]', columns='Replicate', values='Value') #Reshapes the data from long format (one row per time × replicate) to wide format where each column is one replicate and each row is a time point — making it easy to calculate mean and SEM across replicates.
        df_pivot.columns = [f"rep{col_counter + i}" for i in range(df_pivot.shape[1])] #Renames the columns to have unique names across all DataFrames. For example, if the first file has 3 replicates, they will be named rep0, rep1, rep2. If the second file also has 3 replicates, they will be named rep3, rep4, rep5, and so on. This ensures that when we later concatenate or combine these DataFrames, I don't have duplicate column names.
        col_counter += df_pivot.shape[1] #Keeps track of how many replicate columns we've added so far, so that the next set of replicates gets the correct numbering. For example, if the first file had 3 replicates, col_counter would be 3 after processing it, so the next file's replicates would start from rep3.
        all_vals.append(df_pivot) #Adds this pivoted DataFrame to the list of all values for this time point. Each DataFrame in all_vals has the same structure: index is time, columns are replicates, and values are G(t) for the cross-correlation channel.

    all_times   = np.unique(np.concatenate([df.index.values for df in all_vals])) #Gets the time axis from each file's pivoted DataFrame (now stored as the index), concatenates them all together, then removes duplicates with np.unique — same approach as in process_files(). Gives a single sorted array of every unique time point across all files at this timepoint.
    combined_df = pd.DataFrame(index=all_times) #Creates an empty DataFrame using the combined time axis as the index — the scaffold that all replicates from all files will be mapped onto.
    for df in all_vals:
        for col in df.columns:
            combined_df[col] = np.interp(all_times, df.index.values, df[col].values) #Interpolates that replicate's values onto the common time axis and adds it as a new column in combined_df. If a file was measured at slightly different time points than another, interpolation fills in the gaps so all replicates share exactly the same time axis and can be averaged together.

    mean_vals = combined_df.mean(axis=1) #Calculates the mean across all replicate columns at each time point 
    sem_vals  = combined_df.sem(axis=1) #Calculates the standard error of the mean (SEM) across all replicate columns at each time point. This will be used to create the shaded error region around the mean curve in the plot.
    color     = cmap_blues(norm_custom(t)) #Gets a color from the reversed Blues colormap based on the time point t, using the custom normalization to ensure that 0 min is light and 25 min is dark without 35 min being too blue.
    ax_c.plot(all_times, mean_vals, color=color, lw=1.5, solid_capstyle='round', label=f"{t} min") #Plots the mean G(t) curve for this time point on the ax_c subplot, using the color determined by the colormap. The label for the legend is set to the time point (e.g. "0 min", "5 min", etc.).
    ax_c.fill_between(all_times, mean_vals - sem_vals, mean_vals + sem_vals, color=color, alpha=0.3) #Adds a shaded region around the mean curve to represent the SEM. The area between (mean - SEM) and (mean + SEM) is filled with the same color but with some transparency (alpha=0.3) so it doesn't overpower the mean line.

no_tev_files_35 = time_groups_no_tev.get(35, []) #Gets the list of No TEV files for the 35 min time point from the time_groups_no_tev dictionary. If there are no files for 35 min, it returns an empty list. This is because in Panel C we want to plot the No TEV data for 35 min as a separate curve, so we need to specifically access those files.
all_no_tev_vals = []
col_counter_no  = 0
for fp in no_tev_files_35:
    df       = pd.read_excel(fp)
    df_ch    = df[df['Channel'].str.startswith('Fit Channel 1 -> 2')]
    df_pivot = df_ch.pivot(index='Time [ms]', columns='Replicate', values='Value')
    df_pivot.columns = [f"rep{col_counter_no + i}" for i in range(df_pivot.shape[1])]
    col_counter_no += df_pivot.shape[1]
    all_no_tev_vals.append(df_pivot) #Block  It loops through the no TEV files (currently 35 min, or 0 min if you made that change), reads each file, filters to cross-correlation only, pivots to wide format with each replicate as a column, gives each replicate a unique name using col_counter_no to avoid clashes across files, then appends to all_no_tev_vals

if all_no_tev_vals:
    all_times_no = np.unique(np.concatenate([df.index.values for df in all_no_tev_vals]))
    combined_no  = pd.DataFrame(index=all_times_no)
    for df in all_no_tev_vals:
        for col in df.columns:
            combined_no[col] = np.interp(all_times_no, df.index.values, df[col].values)
    mean_no = combined_no.mean(axis=1)
    sem_no  = combined_no.sem(axis=1)
    ax_c.plot(all_times_no, mean_no, color='#006400', lw=1.5, label='No TEV 35 min')
    ax_c.fill_between(all_times_no, mean_no - sem_no, mean_no + sem_no, color='#006400', alpha=0.3) #It takes all the no TEV replicate data, aligns it to a common time axis, calculates the mean and SEM across all replicates, and plots it as a single green reference curve with a shaded SEM band on Panel C.

ax_c.set_xscale('log')
ax_c.set_xlabel('Time [ms]')
ax_c.set_ylabel('G(t)')
ax_c.set_yticks([0.0000, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025])
ax_c.set_ylim(0, 0.0025)
ax_c.xaxis.set_major_locator(LogLocator(base=10.0))
ax_c.xaxis.set_minor_locator(NullLocator())
ax_c.yaxis.set_minor_locator(NullLocator())
ax_c.tick_params(axis='x', which='major', direction='out', length=4, width=1.2, bottom=True)
ax_c.tick_params(axis='y', which='major', direction='out', length=4, width=1.2, left=True)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.grid(False)
ax_c.legend(loc='center right', bbox_to_anchor=(0.98, 0.5), frameon=False,
            edgecolor='black', facecolor='white', fontsize=8) #axis plotting

# ============================================================================
# PANEL D
# ============================================================================
def compute_start_values(tg):
    # First find the global minimum time point across ALL files in this group
    global_min_time = np.inf
    for t in times_sorted:
        for fp in tg.get(t, []):
            df = pd.read_excel(fp)
            df_ch = df[df['Channel'].str.startswith('Fit Channel 1 -> 2')]
            if df_ch.empty:
                continue
            global_min_time = min(global_min_time, df_ch['Time [ms]'].min()) #Find the earliest Time [ms] value across all files — e.g. 0.001 ms, For every replicate in every file, go to that time point and read off the corresponding Fit Channel 1 -> 2 value — i.e. the G(t) amplitude at the start of the correlation curve, That value is essentially G(0) — the cross-correlation amplitude, which reflects how many molecules are moving together

    means, sems = [], []
    for t in times_sorted:
        all_v = []
        for fp in tg.get(t, []):
            df = pd.read_excel(fp)
            df_ch = df[df['Channel'].str.startswith('Fit Channel 1 -> 2')]
            if df_ch.empty:
                continue
            for rep in df_ch['Replicate'].unique():
                df_rep = df_ch[df_ch['Replicate'] == rep].sort_values('Time [ms]')
                if df_rep.empty:
                    continue
                # Interpolate to get value at the global minimum time point
                val = np.interp(global_min_time, 
                                df_rep['Time [ms]'].values, 
                                df_rep['Value'].values)
                all_v.append(val)
        if all_v:
            means.append(np.mean(all_v))
            sems.append(np.std(all_v, ddof=1) / np.sqrt(len(all_v)))
        else:
            means.append(np.nan)
            sems.append(np.nan)
    return means, sems #This block loops through every experimental timepoint (0, 5, 10... 35 min), collects the G(0) value from every replicate across all files at that timepoint, then calculates the mean and SEM from those pooled values. It returns two lists — one mean and one SEM per timepoint — which become the single dot and error bars plotted for each timepoint in Panel D.

means,        sems        = compute_start_values(time_groups)
means_no_tev, sems_no_tev = compute_start_values(time_groups_no_tev) #used to plot the sem and mean in panel D later.

cmap_d   = cm.get_cmap('Blues')
norm_d   = Normalize(vmin=min(times_sorted), vmax=max(times_sorted))
colors_d = [cmap_d(1 - norm_d(t) * 0.7) for t in times_sorted]
offset   = 0.6 #Sets blues again so they arent to dark or blue, and then offset sets gap between the dots on graph.




for t, m, s, c in zip(times_sorted, means, sems, colors_d):
    ax_d.errorbar(t - offset, m, yerr=s, fmt='o', color=c,
                  ecolor='black', elinewidth=1.5, capsize=4, markersize=5,
                  zorder=2, label='TEV' if t == times_sorted[0] else "") #Loops through each timepoint simultaneously pulling the time (t), mean (m), SEM (s), and colour (c) from their respective lists using zip.Loops through each timepoint simultaneously pulling the time (t), mean (m), SEM (s), and colour (c) from their respective lists using zip.For each timepoint it plots one error bar dot on Panel D:

for t, m, s in zip(times_sorted, means_no_tev, sems_no_tev):
    ax_d.errorbar(t + offset, m, yerr=s, fmt='s', color='#006400',
                  ecolor='black', elinewidth=1.5, capsize=4, markersize=4,
                  zorder=2, label='No TEV' if t == times_sorted[0] else "") #same as above for no_tev

def exp_decay(x, a, b, c): #Defines the mathematical function for an exponential decay, where a is the initial amplitude, b is the decay rate, and c is the offset. This function will be fitted to the TEV data points in Panel D to see if they follow an exponential decay pattern over time.
    return a * np.exp(-b * x) + c # The function takes an array of time points (x) and the parameters a, b, c, and returns the corresponding G(0) values according to the exponential decay model. When we fit this function to the TEV data, we will get estimates for a, b, and c that best describe how G(0) changes over time with TEV treatment.

x, y = np.array(times_sorted), np.array(means)
mask = ~np.isnan(y) #converts timepoints and means to numpy, eand excludes any datapoints that are NaN. This is important because if there are any timepoints where we couldn't calculate a mean (e.g. no data), those will be NaN and we don't want to include them in the curve fitting process as it would cause errors.
try:
    params, _ = curve_fit(exp_decay, x[mask], y[mask], maxfev=10000) #Fits the exponential decay function to the real data points only (using the mask). params stores the best fit values for a, b, c. _ discards the covariance matrix. maxfev=10000 allows up to 10000 iterations to find a solution before giving up.
    x_smooth  = np.linspace(min(x[mask]), max(x[mask]), 200) #Creates 200 evenly spaced x values across the data range — gives a smooth continuous curve rather than just connecting the data points.
    ax_d.plot(x_smooth, exp_decay(x_smooth, *params), color='#1f77b4', lw=1.5, label='TEV exp fit')
except RuntimeError:
    print("Exp decay fit did not converge")

y_no    = np.array(means_no_tev)
mask_no = ~np.isnan(y_no)
if mask_no.any():
    def zeroth_order(x, c):
        return np.full_like(x, c, dtype=float)

    try:
        params_no, _ = curve_fit(zeroth_order, x[mask_no], y_no[mask_no])
        x_smooth_no  = np.linspace(min(x[mask_no]), max(x[mask_no]), 200)
        ax_d.plot(x_smooth_no, zeroth_order(x_smooth_no, *params_no),
                  color='#006400', lw=1.5, linestyle='-', label='No TEV fit')
    except RuntimeError:
        print("Zeroth order fit did not converge") #same as tev above except fitting a zeroth order function which is just a flat line at the mean value, to see if no TEV data is consistent with no change over time.

ax_d.set_xlabel("Time (min)")
ax_d.set_ylabel("G(t)")
ax_d.set_yticks([0.0000, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025])
ax_d.set_ylim(0, 0.0025)
ax_d.set_xticks(times_sorted)
ax_d.set_xlim(min(times_sorted) - 2, max(times_sorted) + 2)
ax_d.set_ylim(bottom=0)
ax_d.xaxis.set_minor_locator(NullLocator())
ax_d.yaxis.set_minor_locator(NullLocator())
ax_d.tick_params(axis='both', which='major', direction='out', length=4, width=1.2, bottom=True, left=True)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.grid(False)
ax_d.legend(frameon=False, facecolor='white', edgecolor='black', fontsize=8) #axis plotting

# ============================================================================
# SHOW
# ============================================================================
plt.show()


# %%
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "Figure4.svg")
fig.savefig(out_path, format='svg', dpi=300)
print(f"Saved to: {out_path}")


