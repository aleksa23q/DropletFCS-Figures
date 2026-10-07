# Figure 1: DIB fluorescence images with line profiles
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure
from scipy.ndimage import binary_fill_holes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

image_1_path = "/Users/aleksalakic/Desktop/Figure1B_Oct/250305_dibbot_oil_dendra_1_488.tif"
image_2_path = "/Users/aleksalakic/Desktop/Figure1B_Oct/241014_dibbot_OL_dendra_10_488.tif"
image_3_path = "/Users/aleksalakic/Desktop/Figure1B_Oct/250305_dibbot_peglipid_dendra_5_488.tif"
output_folder = "/Users/aleksalakic/Desktop/Figure1B_Oct/outputs/"
os.makedirs(output_folder, exist_ok=True)

fov_um = 665.6
scale_bar_um = 50
crop_half_um = 165.0

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


def find_droplet_centre(image):
    smooth = filters.gaussian(image.astype(float), sigma=3)
    mask = smooth > filters.threshold_otsu(smooth)
    mask = morphology.remove_small_objects(mask, 500)
    mask = binary_fill_holes(morphology.closing(mask, morphology.disk(5)))
    regions = measure.regionprops(measure.label(mask))
    largest = max(regions, key=lambda region: region.area)
    return largest.centroid


def crop_around(image, cy, cx, half_px):
    # Zero-padding keeps the droplet centred when it sits near the frame edge
    padded = np.pad(image, half_px)
    cy = int(round(cy)) + half_px
    cx = int(round(cx)) + half_px
    return padded[cy - half_px:cy + half_px, cx - half_px:cx + half_px].copy()


images = [io.imread(path) for path in (image_1_path, image_2_path, image_3_path)]

# All images share the same field of view, so pixel size follows from image width
pixel_sizes_um = [fov_um / image.shape[1] for image in images]
for i, (image, pixel_size) in enumerate(zip(images, pixel_sizes_um)):
    print(f"Image {i + 1}: width = {image.shape[1]} px, pixel size = {pixel_size:.2f} um")

crops = []
for image, pixel_size in zip(images, pixel_sizes_um):
    cy, cx = find_droplet_centre(image)
    half_px = int(round(crop_half_um / pixel_size))
    crops.append(crop_around(image, cy, cx, half_px))
    print(f"Droplet centre (y, x) = ({cy:.0f}, {cx:.0f}), crop half-width = {half_px} px")

line_rows = [c.shape[0] // 2 for c in crops]
norm_maxes = [bit_depth_max(c) for c in crops]
linescans = [crops[i][line_rows[i], :] / norm_maxes[i] for i in range(3)]
x_um = [(np.arange(c.shape[1]) - c.shape[1] / 2 + 0.5) * pixel_sizes_um[i]
        for i, c in enumerate(crops)]

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
    ax_line.plot(x_um[col], linescans[col], color="black", linewidth=0.6)
    ax_line.set_ylim(0, y_max)
    ax_line.set_xlim(-crop_half_um, crop_half_um)
    ax_line.set_xlabel("Position (\u00b5m)", fontsize=8)
    if col == 0:
        ax_line.set_ylabel("Normalised intensity", fontsize=8)
    else:
        ax_line.set_yticklabels([])
    ax_line.tick_params(labelsize=7, direction="in", top=True, right=True)

fig.savefig(f"{output_folder}23apr_line_profiles.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(f"{output_folder}23apr_line_profiles.svg", bbox_inches="tight", facecolor="white")
print("Saved.")
plt.show()