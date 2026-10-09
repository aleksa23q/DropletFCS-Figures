# =============================================================================
# Figure S1: mCherry (647) DIB fluorescence images with line profiles
#            -- size-matched droplets
# -----------------------------------------------------------------------------
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure
from scipy.ndimage import binary_fill_holes, median_filter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# ----------------------------- CONFIG ----------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = "/Users/aleksalakic/Desktop/FCS_current/Figure1_23-04-26/Fig1C_analysis/Extracted_channels"

image_files = [
    "240628_dibbot_oil_mCherry_2_647.tif",        # Oil
    "241014_dibbot_OL_mcherry_3_647.tif",         # Oil + lipid
    "250305_dibbot_peglipid_mCherry_3_647.tif",   # Oil + PEGylated lipid
]
labels = ["Oil", "Oil + lipid", "Oil + PEGylated lipid"]

# Copies of the three raw TIFFs sit next to this script, so the folder is
# self-contained; fall back to the original Extracted_channels folder.
image_paths = [os.path.join(HERE, f) if os.path.exists(os.path.join(HERE, f))
               else os.path.join(SRC, f) for f in image_files]

output_folder = HERE

FOV_UM        = 512 * 1.3   # 665.6 um field of view (= 1024 * 0.65)
CROP_HALF_UM  = 225.0       # half-width of every crop, in um (identical scale)
scale_bar_um  = 50
PROFILE_MEDIAN_PX = 3       # hot-pixel rejection along the line profile
NORMALISATION = "profile"   # "profile" (per-profile max) or "bitdepth"

# Optional manual override of the droplet centre, per image, as (cy, cx) in
# pixels. Leave as None to use the automatically segmented centroid.
centre_override = [None, None, None]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.dpi": 300,
})
cm = 1 / 2.54


# --------------------------- helpers -----------------------------------------
def bit_depth_max(image):
    """Original helper: full-scale value for the image's (sub-byte) bit depth."""
    dtype_max = np.iinfo(image.dtype).max
    observed_max = image.max()
    if image.dtype == np.uint16:
        if observed_max <= 4095:
            return 4095
        if observed_max <= 16383:
            return 16383
    return dtype_max


def find_droplet(image):
    """Segment the largest bright object; return (cy, cx, r_px, background)."""
    img = image.astype(float)
    h, w = img.shape
    smooth = filters.gaussian(img, sigma=3)
    mask = smooth > filters.threshold_otsu(smooth)
    mask = morphology.remove_small_objects(mask, 500)
    mask = binary_fill_holes(morphology.closing(mask, morphology.disk(5)))
    region = sorted(measure.regionprops(measure.label(mask)),
                    key=lambda p: -p.area)[0]
    cy, cx = region.centroid
    r_px = np.sqrt(region.area / np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / r_px
    outside = img[rr > 1.20]
    background = float(np.median(outside)) if outside.size else 0.0
    return cy, cx, r_px, background


def make_panel(path, centre=None):
    """Load an image, centre-crop it to CROP_HALF_UM and extract the profile."""
    image = io.imread(path)
    h, w = image.shape
    px_um = FOV_UM / w
    cy, cx, r_px, background = find_droplet(image)
    if centre is not None:
        cy, cx = centre

    half_px = int(round(CROP_HALF_UM / px_um))
    y0, x0 = int(round(cy)) - half_px, int(round(cx)) - half_px
    y1, x1 = y0 + 2 * half_px, x0 + 2 * half_px
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        raise ValueError(f"{os.path.basename(path)}: crop of +/-{CROP_HALF_UM} um "
                         f"falls outside the frame; reduce CROP_HALF_UM.")
    crop = image[y0:y1, x0:x1].copy()

    line_row = crop.shape[0] // 2
    raw = crop[line_row, :].astype(float)
    profile = median_filter(raw, size=PROFILE_MEDIAN_PX)

    if NORMALISATION == "bitdepth":
        profile = profile / bit_depth_max(crop)
    else:
        profile = profile - background
        profile = profile / profile.max()

    x_um = (np.arange(crop.shape[1]) - crop.shape[1] / 2 + 0.5) * px_um
    return dict(name=os.path.basename(path), crop=crop, px_um=px_um,
                line_row=line_row, profile=profile, x_um=x_um,
                d_um=2 * r_px * px_um, cy=cy, cx=cx, bits=str(image.dtype),
                background=background)


# --------------------------- build panels ------------------------------------
panels = [make_panel(p, c) for p, c in zip(image_paths, centre_override)]

print(f"Normalisation mode: {NORMALISATION}")
print(f"{'image':<44s} {'cond':<22s} {'bits':>7s} {'px/um':>6s} "
      f"{'diam(um)':>9s} {'centre(y,x)':>14s}")
for p, lab in zip(panels, labels):
    centre_str = "(%.0f,%.0f)" % (p['cy'], p['cx'])
    print(f"{p['name']:<44s} {lab:<22s} {p['bits']:>7s} {p['px_um']:>6.2f} "
          f"{p['d_um']:>9.1f} {centre_str:>14s}")
spread = (max(p['d_um'] for p in panels) / min(p['d_um'] for p in panels) - 1) * 100
print(f"Diameter spread across the three panels: {spread:.1f}%")


# --------------------------- figure ------------------------------------------
fig_width, fig_height = 18 * cm, 9.5 * cm
fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
gs = GridSpec(2, 3, figure=fig, height_ratios=[4, 1.2], hspace=0.05, wspace=0.05)

y_max = max(p['profile'].max() for p in panels) * 1.1
x_lim = CROP_HALF_UM

for col, p in enumerate(panels):
    crop = p['crop']
    img_h, img_w = crop.shape

    ax_img = fig.add_subplot(gs[0, col])
    ax_img.imshow(crop, cmap="gray", aspect="equal",
                  vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99.8))
    ax_img.axhline(p['line_row'], color="yellow", linewidth=1.2,
                   linestyle="--", alpha=0.9)
    ax_img.axis("off")

    bar_length_px = scale_bar_um / p['px_um']
    bar_thickness_px = max(img_h * 0.012, 3)
    margin_px = img_w * 0.04
    ax_img.add_patch(Rectangle(
        (img_w - bar_length_px - margin_px, img_h - margin_px - bar_thickness_px),
        bar_length_px, bar_thickness_px, facecolor="white", edgecolor="none"))

    ax_line = fig.add_subplot(gs[1, col])
    ax_line.plot(p['x_um'], p['profile'], color="black", linewidth=0.8)
    ax_line.set_ylim(0, y_max)
    ax_line.set_xlim(-x_lim, x_lim)
    # explicit ticks kept inside the axes so adjacent panels' edge labels
    # do not run into each other
    ax_line.set_xticks([-150, 0, 150])
    ax_line.set_yticks([0, 0.5, 1.0] if NORMALISATION == "profile"
                       else np.round(np.linspace(0, y_max, 3), 2))
    ax_line.set_xlabel("Position (\u00b5m)")
    if col == 0:
        ax_line.set_ylabel("Normalised intensity")
    else:
        ax_line.set_yticklabels([])
    ax_line.tick_params(direction="in", top=True, right=True)

stem = f"figureS1_line_profiles_{NORMALISATION}norm"
fig.savefig(os.path.join(output_folder, stem + ".png"), dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(output_folder, stem + ".svg"),
            bbox_inches="tight", facecolor="white")
print(f"Saved {stem}.png / .svg")
