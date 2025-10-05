# segmentation_to_instances.py
# Segment a microscopy image and save each instance as:
#   - a cropped binary mask .npy (uint8 with 0/1)
#   - a JSON metadata file with global coordinates (bbox, image size, centroid)
# The output folder is cleaned before saving.
# All code and comments are ASCII only.

import os
import json
import shutil
import tkinter as tk
from tkinter import filedialog
import numpy as np

from skimage import io
from skimage.color import gray2rgb
from skimage.measure import label, regionprops

try:
    from aicsimageio import AICSImage
    HAS_AICS = True
except Exception:
    HAS_AICS = False

from cellpose import models


def select_image_file():
    """Ask user to pick an image file."""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select microscopy image",
        filetypes=[("Images", "*.czi *.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
    )

def select_output_folder():
    """Ask user to pick an output folder for instances."""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title="Select output folder for instances")

def cleanup_output_folder(folder):
    """Delete previous outputs to avoid conflicts."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def load_image_rgb(image_path):
    """
    Load image and return an RGB array (H, W, 3).
    Supports CZI via AICSImage if installed, otherwise uses skimage.io for common formats.
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext == ".czi":
        if not HAS_AICS:
            raise RuntimeError("AICSImage not available. Install aicsimageio to read CZI.")
        img = AICSImage(image_path)
        try:
            if "C" in img.dims.order and img.dims.get("C", 1) >= 3:
                r = img.get_image_data("YX", S=0, T=0, C=0)
                g = img.get_image_data("YX", S=0, T=0, C=1)
                b = img.get_image_data("YX", S=0, T=0, C=2)
                rgb = np.stack([r, g, b], axis=-1)
            else:
                yx = img.get_image_data("YX", S=0, T=0, C=0)
                rgb = gray2rgb(yx)
        except Exception:
            yx = img.get_image_data("YX", S=0, T=0, C=0)
            rgb = gray2rgb(yx)
        return rgb
    else:
        arr = io.imread(image_path)
        if arr.ndim == 2:
            return gray2rgb(arr)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return arr[:, :, :3]
        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr
        first = arr[..., 0]
        return gray2rgb(first)


def main():
    print("=== Segmentation -> per-instance .npy + .json (global placement) ===")

    image_path = select_image_file()
    if not image_path:
        print("No image selected. Exiting.")
        return

    out_dir = select_output_folder()
    if not out_dir:
        print("No output folder selected. Exiting.")
        return

    cleanup_output_folder(out_dir)
    print("Output folder cleaned:", out_dir)

    print("Loading image ...")
    rgb = load_image_rgb(image_path)
    H, W = rgb.shape[:2]
    print("Image size: {} x {}".format(W, H))

    print("Running Cellpose segmentation (CPU) ...")
    model = models.CellposeModel(gpu=False)
    result = model.eval(rgb, diameter=None, channels=[0, 0])
    masks = result[0] if isinstance(result, (list, tuple)) else result
    print("Segmentation done.")

    labeled = label(masks)
    regions = regionprops(labeled)
    print("Found {} objects".format(len(regions)))

    saved = 0
    for r in regions:
        if r.area < 10:
            continue

        minr, minc, maxr, maxc = r.bbox
        crop = (labeled[minr:maxr, minc:maxc] == r.label).astype(np.uint8)

        inst_path = os.path.join(out_dir, "inst_{:04d}.npy".format(saved))
        np.save(inst_path, crop)

        meta = {
            "instance_id": saved,
            "bbox": [int(minc), int(minr), int(maxc - minc), int(maxr - minr)],  # [x, y, width, height]
            "image_shape": [int(H), int(W)],  # [H, W]
            "area_px": int(r.area),
            "centroid_xy": [float(r.centroid[1]), float(r.centroid[0])],  # (x, y)
            "source_image": os.path.basename(image_path)
        }
        with open(os.path.join(out_dir, "inst_{:04d}_meta.json".format(saved)), "w") as jf:
            json.dump(meta, jf, indent=2)

        saved += 1

    print("Saved {} instances to: {}".format(saved, out_dir))
    print("Each instance has .npy mask (cropped) and .json with bbox and image dimensions.")


if __name__ == "__main__":
    main()
