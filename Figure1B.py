# Figure 1: DIB fluorescence images with line profiles
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

image_1_path = "/Users/aleksalakic/Desktop/Figure1_22-05-26/Figure1B_analysis/250305_dibbot_oil_dendra_1_488.tif"
image_2_path = "/Users/aleksalakic/Desktop/Figure1_22-05-26/Figure1B_analysis/241014_dibbot_OL_dendra_2_488.tif"
image_3_path = "/Users/aleksalakic/Desktop/Figure1_22-05-26/Figure1B_analysis/241011_dibbot_peglipid_dendra+mCherry_6_488.tif"
output_folder = "/Users/aleksalakic/Desktop/Figure1_22-05-26/Figure1B_analysis/outputs/"
os.makedirs(output_folder, exist_ok=True)

pixel_sizes_um = [1.3, 0.65, 0.65]
scale_bar_um = 50

font = {'family': 'arial', 'weight': 'normal', 'size': 8}
plt.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm = 1 / 2.54


def bit_depth_max(image):
    dtype_max = np.iinfo(image.dtype).max
    observed_max = image.max()
    # Detect common sub-byte-boundary ranges (12-bit, 14-bit) stored as uint16
    if image.dtype == np.uint16:
        if observed_max <= 4095:
            return 4095
        if observed_max <= 16383:
            return 16383
    return dtype_max


image_1 = io.imread(image_1_path)
image_2 = io.imread(image_2_path)
image_3 = io.imread(image_3_path)

cy1, cx1, half1 = 179, 217, 130
cy2, cx2, half2 = 499, 510, 350
cy3, cx3, half3 = 603, 405, 370

crop_1 = image_1[cy1 - half1:cy1 + half1, cx1 - half1:cx1 + half1].copy()
crop_2 = image_2[cy2 - half2:cy2 + half2, cx2 - half2:cx2 + half2].copy()
crop_3 = image_3[cy3 - half3:cy3 + half3, cx3 - half3:cx3 + half3].copy()

crops = [crop_1, crop_2, crop_3]
line_rows = [c.shape[0] // 2 for c in crops]
norm_maxes = [bit_depth_max(c) for c in crops]
linescans = [crops[i][line_rows[i], :] / norm_maxes[i] for i in range(3)]

print("Detected normalisation maxima:")
for i, (c, m) in enumerate(zip(crops, norm_maxes)):
    print(f"  Image {i + 1}: dtype={c.dtype}, observed max={c.max()}, normalised to {m}")

fig_width = 18 * cm
fig_height = 8 * cm
fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
gs = GridSpec(2, 3, figure=fig, height_ratios=[4, 1.2], hspace=0.05, wspace=0.08)

y_max = max(ls.max() for ls in linescans) * 1.1

for col in range(3):
    ax_img = fig.add_subplot(gs[0, col])
    ax_img.imshow(crops[col], cmap="gray", aspect="equal")
    ax_img.axhline(line_rows[col], color="yellow", linewidth=1.0, linestyle="--", alpha=0.9)
    ax_img.axis("off")

    img_h, img_w = crops[col].shape
    bar_length_px = scale_bar_um / pixel_sizes_um[col]
    bar_thickness_px = img_h * 0.015
    margin_px = img_w * 0.05
    bar_x = img_w - bar_length_px - margin_px
    bar_y = img_h - margin_px - bar_thickness_px
    ax_img.add_patch(Rectangle(
        (bar_x, bar_y), bar_length_px, bar_thickness_px,
        facecolor="white", edgecolor="none"
    ))

    ax_line = fig.add_subplot(gs[1, col])
    ax_line.plot(linescans[col], color="black", linewidth=0.6)
    ax_line.set_ylim(0, y_max)
    ax_line.set_xlabel("Pixel position", fontsize=8)
    if col == 0:
        ax_line.set_ylabel("Normalised intensity", fontsize=8)
    else:
        ax_line.set_yticklabels([])
    ax_line.tick_params(labelsize=7, direction="in", top=True, right=True)

fig.savefig(f"{output_folder}23apr_line_profiles.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{output_folder}23apr_line_profiles.svg", bbox_inches="tight", facecolor="white")
print("Saved.")
plt.show()