import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from skimage import io

image_1_path = "/Users/aleksalakic/Desktop/Supp_figure_mcherry_coating/240628_dibbot_oil_mCherry_2_647.tif"
image_2_path = "/Users/aleksalakic/Desktop/Supp_figure_mcherry_coating/241014_dibbot_OL_mcherry_4_647.tif"
image_3_path = "/Users/aleksalakic/Desktop/Supp_figure_mcherry_coating/250305_dibbot_peglipid_mCherry_1_647.tif"
output_folder = "/Users/aleksalakic/Desktop/Supp_figure_mcherry_coating/outputs/"
os.makedirs(output_folder, exist_ok=True)

pixel_sizes_um = [1.3, 0.65, 0.65]
scale_bar_um = 50

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def bit_depth_max(image):
    dtype_max = np.iinfo(image.dtype).max
    observed_max = image.max()
    # Detect 12/14-bit data stored in uint16 containers
    if image.dtype == np.uint16:
        if observed_max <= 4095:
            return 4095
        if observed_max <= 16383:
            return 16383
    return dtype_max


image_1 = io.imread(image_1_path)
image_2 = io.imread(image_2_path)
image_3 = io.imread(image_3_path)

cy1, cx1, half1 = 400, 570, 400
cy2, cx2, half2 = 499, 570, 350
cy3, cx3, half3 = 350, 350, 320

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

cm = 1 / 2.54
fig_width = 18 * cm
fig_height = 12.5 * cm

fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
gs = GridSpec(2, 3, figure=fig, height_ratios=[4, 1.2], hspace=0.05, wspace=0.05)

y_max = max(ls.max() for ls in linescans) * 1.1

for col in range(3):
    ax_img = fig.add_subplot(gs[0, col])
    ax_img.imshow(crops[col], cmap="gray", aspect="equal")
    ax_img.axhline(line_rows[col], color="yellow", linewidth=1.2, linestyle="--", alpha=0.9)
    ax_img.axis("off")

    img_h, img_w = crops[col].shape
    bar_length_px = scale_bar_um / pixel_sizes_um[col]
    bar_thickness_px = max(img_h * 0.012, 3)
    margin_px = img_w * 0.04
    bar_x = img_w - bar_length_px - margin_px
    bar_y = img_h - margin_px - bar_thickness_px

    ax_img.add_patch(Rectangle(
        (bar_x, bar_y), bar_length_px, bar_thickness_px,
        facecolor="white", edgecolor="none"
    ))

    ax_line = fig.add_subplot(gs[1, col])
    ax_line.plot(linescans[col], color="black", linewidth=0.8)
    ax_line.set_ylim(0, y_max)
    ax_line.set_xlim(0, len(linescans[col]) - 1)
    ax_line.set_xlabel("Pixel position")

    if col == 0:
        ax_line.set_ylabel("Normalised intensity")
    else:
        ax_line.set_yticklabels([])

    ax_line.tick_params(direction="in", top=True, right=True)

fig.savefig(f"{output_folder}23apr_line_profiles.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{output_folder}23apr_line_profiles.svg", bbox_inches="tight", facecolor="white")
print("Saved.")
plt.show()