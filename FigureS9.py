#sup figure 9

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from skimage.io import imread
from skimage.morphology import dilation, disk
from skimage.measure import find_contours
from scipy.ndimage import binary_erosion

root_path = open('experimental_data/experiment_index.txt', 'r').readlines()[0].strip()
input_folder = f'{root_path}Supp_figure/'
mask_folder = f'{root_path}Masks/fluoro/'
output_folder = f'{root_path}SuppFig6/'
os.makedirs(output_folder, exist_ok=True)

stacks = [
    '250305_dibbot_oil_dendra_1',
    '241014_dibbot_OL_dendra+mcherry_2',
    '241011_dibbot_peglipid_dendra+mCherry_6',
]
channel = '488'
mask_suffix = '_fluoro_mask.npy'

outer_expansion = 0
inner_offset_fraction = 0.5
inner_offset_min = 0
inner_offset_max = 15
background_dilation = 3

boundary_shade = (0.2, 1.0, 0.2, 0.35)
interior_shade = (1.0, 0.2, 1.0, 0.30)

pixel_size_um = 0.378
scalebar_length_um = 100
scalebar_thickness_px = 6
scalebar_pad_px = 20
scalebar_color = 'white'
scalebar_label = f'{scalebar_length_um} µm'
scalebar_show_label = False

zoom_scalebar_length_um = 25
zoom_size_px = 200
zoom_box_color = 'yellow'
zoom_box_linewidth = 0.8

condition_labels = {
    '250305_dibbot_oil_dendra_1': 'Oil',
    '241014_dibbot_OL_dendra+mcherry_2': 'Oil-Lipid',
    '241011_dibbot_peglipid_dendra+mCherry_6': 'PEG-lipid',
}


def compute_inner_offset(area_px):
    if area_px <= 0:
        return inner_offset_min
    r_eff = np.sqrt(area_px / np.pi)
    offset = int(round(inner_offset_fraction * r_eff))
    return max(inner_offset_min, min(inner_offset_max, offset))


def get_region_masks(binary_mask, outer_px, inner_px):
    outer = dilation(binary_mask, disk(outer_px)) if outer_px > 0 else binary_mask.copy()
    if inner_px > 0:
        inner = binary_erosion(binary_mask, iterations=inner_px)
        if not np.any(inner):
            inner = binary_erosion(binary_mask, iterations=1)
    else:
        inner = binary_mask.copy()
    return outer & ~inner, inner


def get_frame_background(image, union_mask, dilation_px):
    dilated = dilation(union_mask > 0, disk(dilation_px)) if dilation_px > 0 else (union_mask > 0)
    outside = ~dilated
    if not np.any(outside):
        return np.nan
    return float(np.mean(image[outside]))


def add_scalebar(ax, image_shape, length_um, pixel_size_um, thickness_px, pad_px, color, label, show_label):
    length_px = length_um / pixel_size_um
    h, w = image_shape
    x0 = w - pad_px - length_px
    y0 = h - pad_px - thickness_px
    ax.add_patch(patches.Rectangle((x0, y0), length_px, thickness_px,
                                   facecolor=color, edgecolor='none'))
    if show_label:
        ax.text(x0 + length_px / 2, y0 - 4, label,
                color=color, ha='center', va='bottom', fontsize=7)


def pick_zoom_region(mask, image_shape, zoom_size):
    h, w = image_shape
    half = zoom_size // 2
    labels = np.unique(mask)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return max(0, h // 2 - half), max(0, w // 2 - half)

    centre = np.array([h / 2, w / 2])
    best_dist = np.inf
    best_ys, best_xs = None, None
    for lab in labels:
        ys, xs = np.where(mask == lab) if mask.max() > 1 else np.where(mask > 0)
        if len(ys) == 0:
            continue
        d = np.hypot(ys.mean() - centre[0], xs.mean() - centre[1])
        if d < best_dist:
            best_dist = d
            best_ys, best_xs = ys, xs
        if mask.max() <= 1:
            break

    cy, cx = best_ys.mean(), best_xs.mean()
    # angle measured in image coords: +y is down, so 45° below horizontal = (+y, +x)
    angles = np.arctan2(best_ys - cy, best_xs - cx)
    target = np.pi / 4
    idx = np.argmin(np.abs(angles - target))
    by, bx = best_ys[idx], best_xs[idx]

    y0 = max(0, min(h - zoom_size, by - half))
    x0 = max(0, min(w - zoom_size, bx - half))
    return int(y0), int(x0)


panel_a_images = {}
panel_b_overlays = {}
panel_b_masks = {}
panel_c_rows = []

for stack_name in stacks:
    image_path = os.path.join(input_folder, f'{stack_name}_{channel}.tif')
    mask_path = os.path.join(mask_folder, f'{stack_name}{mask_suffix}')

    image = imread(image_path)
    image = np.squeeze(image)
    if image.ndim == 3:
        image = image[0]

    mask_stack = np.load(mask_path)
    mask = mask_stack[0] if mask_stack.ndim == 3 else mask_stack

    panel_a_images[stack_name] = image
    panel_b_masks[stack_name] = mask

    droplet_labels = np.unique(mask)
    droplet_labels = droplet_labels[droplet_labels > 0]
    background = get_frame_background(image, mask, background_dilation)

    boundary_union = np.zeros(image.shape, dtype=bool)
    interior_union = np.zeros(image.shape, dtype=bool)
    boundary_contours = []
    interior_contours = []

    for droplet_id in droplet_labels:
        binary_mask = (mask == droplet_id) if mask.max() > 1 else (mask > 0)
        area_px = int(np.sum(binary_mask))
        if area_px == 0:
            continue
        inner_px = compute_inner_offset(area_px)
        boundary_ring, interior_core = get_region_masks(binary_mask, outer_expansion, inner_px)

        boundary_union |= boundary_ring
        interior_union |= interior_core

        for c in find_contours(binary_mask.astype(float), 0.5):
            boundary_contours.append(c)
        for c in find_contours(interior_core.astype(float), 0.5):
            interior_contours.append(c)

        boundary_pixels = image[boundary_ring]
        interior_pixels = image[interior_core]
        if len(boundary_pixels) == 0 or len(interior_pixels) == 0:
            continue

        panel_c_rows.append({
            'condition': condition_labels[stack_name],
            'droplet_id': int(droplet_id),
            'boundary_max': float(np.max(boundary_pixels)),
            'interior_mean': float(np.mean(interior_pixels)),
            'normalised_intensity': float(np.max(boundary_pixels)) / float(np.mean(interior_pixels)),
        })

        if mask.max() <= 1:
            break

    panel_b_overlays[stack_name] = (boundary_contours, interior_contours,
                                    boundary_union, interior_union)

panel_c = pd.DataFrame(panel_c_rows)

font = {'family': 'arial', 'weight': 'normal', 'size': 8}
plt.rc('font', **font)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300
cm = 1 / 2.54

fig = plt.figure(figsize=(18 * cm, 22 * cm))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.1], hspace=0.25, wspace=0.15,
                      left=0.1, right=0.98, top=0.97, bottom=0.07)

for i, stack_name in enumerate(stacks):
    ax = fig.add_subplot(gs[0, i])
    img = panel_a_images[stack_name]
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(condition_labels[stack_name], fontsize=8)
    add_scalebar(ax, img.shape, scalebar_length_um, pixel_size_um,
                 scalebar_thickness_px, scalebar_pad_px,
                 scalebar_color, scalebar_label, scalebar_show_label)
    ax.axis('off')

zoom_coords = {}
for i, stack_name in enumerate(stacks):
    ax = fig.add_subplot(gs[1, i])
    img = panel_a_images[stack_name]
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)

    boundary_contours, interior_contours, boundary_union, interior_union = panel_b_overlays[stack_name]

    overlay = np.zeros((*img.shape, 4))
    overlay[boundary_union] = boundary_shade
    overlay[interior_union] = interior_shade
    ax.imshow(overlay)

    for c in boundary_contours:
        ax.plot(c[:, 1], c[:, 0], color='red', linestyle=':', linewidth=0.8)
    for c in interior_contours:
        ax.plot(c[:, 1], c[:, 0], color='red', linestyle=':', linewidth=0.8)

    y0, x0 = pick_zoom_region(panel_b_masks[stack_name], img.shape, zoom_size_px)
    zoom_coords[stack_name] = (y0, x0)
    ax.add_patch(patches.Rectangle((x0, y0), zoom_size_px, zoom_size_px,
                                   facecolor='none', edgecolor=zoom_box_color,
                                   linewidth=zoom_box_linewidth))
    add_scalebar(ax, img.shape, scalebar_length_um, pixel_size_um,
                 scalebar_thickness_px, scalebar_pad_px,
                 scalebar_color, scalebar_label, scalebar_show_label)
    ax.axis('off')

for i, stack_name in enumerate(stacks):
    ax = fig.add_subplot(gs[2, i])
    img = panel_a_images[stack_name]
    vmin, vmax = np.percentile(img, [1, 99])
    y0, x0 = zoom_coords[stack_name]
    y1, x1 = y0 + zoom_size_px, x0 + zoom_size_px

    img_crop = img[y0:y1, x0:x1]
    ax.imshow(img_crop, cmap='gray', vmin=vmin, vmax=vmax)

    boundary_contours, interior_contours, boundary_union, interior_union = panel_b_overlays[stack_name]
    overlay_crop = np.zeros((zoom_size_px, zoom_size_px, 4))
    overlay_crop[boundary_union[y0:y1, x0:x1]] = boundary_shade
    overlay_crop[interior_union[y0:y1, x0:x1]] = interior_shade
    ax.imshow(overlay_crop)

    for c in boundary_contours:
        cy, cx = c[:, 0] - y0, c[:, 1] - x0
        in_box = (cy >= 0) & (cy < zoom_size_px) & (cx >= 0) & (cx < zoom_size_px)
        if np.any(in_box):
            ax.plot(cx, cy, color='red', linestyle=':', linewidth=0.8)
    for c in interior_contours:
        cy, cx = c[:, 0] - y0, c[:, 1] - x0
        in_box = (cy >= 0) & (cy < zoom_size_px) & (cx >= 0) & (cx < zoom_size_px)
        if np.any(in_box):
            ax.plot(cx, cy, color='red', linestyle=':', linewidth=0.8)

    ax.set_xlim(0, zoom_size_px)
    ax.set_ylim(zoom_size_px, 0)
    for spine in ax.spines.values():
        spine.set_edgecolor(zoom_box_color)
        spine.set_linewidth(zoom_box_linewidth)
    add_scalebar(ax, (zoom_size_px, zoom_size_px), zoom_scalebar_length_um, pixel_size_um,
                 scalebar_thickness_px, scalebar_pad_px,
                 scalebar_color, f'{zoom_scalebar_length_um} µm', scalebar_show_label)
    ax.set_xticks([])
    ax.set_yticks([])

ax_c = fig.add_subplot(gs[3, :])
order = ['Oil', 'Oil-Lipid', 'PEG-lipid']
sns.barplot(
    data=panel_c, x='condition', y='normalised_intensity', order=order,
    palette='Set2', capsize=0.15, errwidth=1, ax=ax_c, edgecolor='white'
)
sns.stripplot(
    data=panel_c, x='condition', y='normalised_intensity', order=order,
    palette='Set2', edgecolor='white', linewidth=1, s=5, ax=ax_c
)
ax_c.set_ylabel('Max Boundary / Mean Interior Intensity')
ax_c.set_xlabel('')
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)

panel_letter_x = 0.02
fig.text(panel_letter_x, 0.96, 'A', fontsize=12, fontweight='bold', va='top', ha='left')
fig.text(panel_letter_x, 0.73, 'B', fontsize=12, fontweight='bold', va='top', ha='left')
fig.text(panel_letter_x, 0.27, 'C', fontsize=12, fontweight='bold', va='top', ha='left')

output_path = os.path.join(output_folder, 'supp_figure_boundary_interior.svg')
plt.savefig(output_path, bbox_inches='tight', dpi=300, format='svg')
plt.savefig(output_path.replace('.svg', '.png'), bbox_inches='tight', dpi=300)

panel_c.to_csv(os.path.join(output_folder, 'supp_figure_panel_c_data.csv'), index=False)
print(f"Saved figure to {output_path}")
print(f"Panel C: {len(panel_c)} droplets across {panel_c['condition'].nunique()} conditions")
print(panel_c.groupby('condition')['normalised_intensity'].agg(['mean', 'std', 'count']))