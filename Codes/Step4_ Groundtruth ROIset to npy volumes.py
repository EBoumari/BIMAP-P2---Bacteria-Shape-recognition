# gt_zip_to_volumes_fixed.py
# Convert ImageJ ROI ZIP sets (GT) + a CZI z-stack into per-instance 3D volumes
# MATCHING the predicted format:
#   - CROPPED per-bacterium (small files), with offset_xy_global_px recorded,
#   - voxel_size_xy_um / voxel_size_z_um in MICRONS (unit-guarded),
#   - z_min_um so that occupied Z-mid maps to Z = 0 microns,
#   - saves: vol_XXXX.npy (uint8, Z,Y,X) and vol_XXXX_meta.json.
#
# ASCII-only comments. Requires: numpy, scikit-image, scipy, czifile.

import os
import json
import struct
import zipfile
import shutil
import tkinter as tk
from tkinter import filedialog
from xml.etree import ElementTree as ET

import numpy as np
from skimage.draw import polygon as sk_polygon
from scipy.ndimage import binary_fill_holes
from czifile import CziFile

# ---------------------------
# UI helpers
# ---------------------------
def select_file(title, patterns):
    root = tk.Tk(); root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=patterns)

def select_folder(title):
    root = tk.Tk(); root.withdraw()
    return filedialog.askdirectory(title=title)

def cleanup_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# ---------------------------
# Unit helpers (ALWAYS MICRONS)
# ---------------------------
def to_um(val, unit):
    """Convert value with unit to microns. Accepts m, mm, um/µm/micron, nm."""
    if val is None:
        return None
    v = float(val)
    u = (unit or "").strip().lower()
    if "µ" in u: u = u.replace("µ", "u")
    if "micron" in u or u == "um":
        return v
    if u == "nm":
        return v * 1e-3
    if u == "mm":
        return v * 1e3
    if u == "m":
        return v * 1e6
    # Unknown -> assume already microns
    return v

def guard_um(v):
    """
    Auto-fix if a value that SHOULD be microns is accidentally stored as meters/nm.
    """
    if v is None:
        return None
    v = float(v)
    # Values expected here:
    #   vxy ~ 0.01 .. 1.0 microns
    #   vz  ~ similar range
    #   z_min_um ~ few microns (not e-6)
    if 0 < v < 1e-4:       # looks like meters -> microns
        return v * 1e6
    if v > 1e4:            # looks like nanometers -> microns
        return v / 1e3
    return v

# ---------------------------
# CZI loader and scaling (MICRONS)
# ---------------------------
def load_czi_shape_and_scaling(czi_path):
    """
    Return (Z, Y, X) and (vxy_um, vz_um) from CZI.
    Pick a single channel/time/scene; parse XML Scaling in microns.
    """
    with CziFile(czi_path) as czi:
        arr = czi.asarray()
        shape = arr.shape
        ndim = arr.ndim

        # Choose the two largest dims as Y and X, another >1 as Z
        dims_sorted = sorted([(i, s) for i, s in enumerate(shape)], key=lambda x: x[1], reverse=True)
        y_ax = dims_sorted[0][0]
        x_ax = dims_sorted[1][0]
        z_ax = None
        for i, s in dims_sorted[2:]:
            if s > 1:
                z_ax = i
                break

        # Index others at 0, keep Z/Y/X as full; squeeze to 3D
        idx = []
        for i in range(ndim):
            if i in (z_ax, y_ax, x_ax):
                idx.append(slice(None))
            else:
                idx.append(0)
        sub = np.squeeze(arr[tuple(idx)])
        if sub.ndim != 3:
            raise RuntimeError(f"Unexpected CZI subarray ndim={sub.ndim}")

        # Reorder to (Z,Y,X)
        axes_sizes = sub.shape
        order = np.argsort(axes_sizes)[::-1]  # largest first -> Y,X, then Z
        Y_i = order[0]; X_i = order[1]
        Z_i = [a for a in (0, 1, 2) if a not in (Y_i, X_i)][0]
        zyx = np.moveaxis(sub, (Z_i, Y_i, X_i), (0, 1, 2))
        Z, Y, X = map(int, zyx.shape)

        # Parse scaling (to microns)
        vxy_um = 0.03225
        vz_um = 0.03225
        try:
            xml = czi.metadata()
            root = ET.fromstring(xml)

            def find_val(id_name):
                for dist in root.findall(".//Scaling//Items//Distance"):
                    if dist.get("Id") == id_name:
                        v = dist.find("Value")
                        u = dist.find("DefaultUnitFormat")
                        unit = u.text if (u is not None and u.text) else None
                        if unit is None:
                            uu = dist.find("Unit")
                            unit = uu.text if (uu is not None and uu.text) else None
                        val = float(v.text) if v is not None else None
                        return val, unit
                return None, None

            vx_val, vx_unit = find_val("X")
            vy_val, vy_unit = find_val("Y")
            vz_val, vz_unit = find_val("Z")

            vx_um = to_um(vx_val, vx_unit) if vx_val is not None else None
            vy_um = to_um(vy_val, vy_unit) if vy_val is not None else None
            vz_um_parsed = to_um(vz_val, vz_unit) if vz_val is not None else None

            if vx_um is not None and vy_um is not None:
                vxy_um = 0.5 * (vx_um + vy_um)
            elif vx_um is not None:
                vxy_um = vx_um
            elif vy_um is not None:
                vxy_um = vy_um

            if vz_um_parsed is not None:
                vz_um = vz_um_parsed
        except Exception as e:
            print("Warning: could not parse CZI scaling, using defaults. Error:", e)

        # Final guard
        vxy_um = guard_um(vxy_um) or 0.03225
        vz_um  = guard_um(vz_um)  or vxy_um

        print(f"CZI shape: Z={Z}, Y={Y}, X={X} | vxy_um={vxy_um:.6f}, vz_um={vz_um:.6f}")
        return (Z, Y, X), float(vxy_um), float(vz_um)

# ---------------------------
# ROI parser (ImageJ .roi, polygon with slice)
# ---------------------------
def parse_imagej_roi_bytes(data):
    """
    Minimal ImageJ ROI parser for polygon ROIs with slice index.
    Returns dict: coords (N,2) absolute X,Y (int), slice (1-based int or None)
    """
    if not (len(data) >= 64 and data[0:4] == b'Iout'):
        raise ValueError("Not an ImageJ ROI file")

    def get_short(off):  return struct.unpack_from('>h', data, off)[0]
    def get_ushort(off): return struct.unpack_from('>H', data, off)[0]
    def get_int(off):    return struct.unpack_from('>i', data, off)[0]

    top  = get_short(8)
    left = get_short(10)
    n = get_ushort(16)
    if n == 0:
        n = get_int(18)

    xs = [struct.unpack_from('>h', data, 64 + 2*i)[0] for i in range(n)]
    ys = [struct.unpack_from('>h', data, 64 + 2*n + 2*i)[0] for i in range(n)]
    coords = np.column_stack([xs, ys]).astype(np.int32)
    coords[:, 0] += left
    coords[:, 1] += top

    hdr2 = get_int(60)
    slice_idx = None
    if hdr2 > 0 and hdr2 + 12 <= len(data):
        slice_idx = get_int(hdr2 + 8)

    return {"coords": coords, "slice": slice_idx}

# ---------------------------
# Build GT volume for ONE ZIP (full-frame first, then crop)
# ---------------------------
def build_gt_volume_from_zip(zip_path, zyx_shape):
    """
    Create a binary full-frame volume (Z,Y,X) from one ROI ZIP set.
    Fill polygons per annotated Z slice and close 2D holes slice-wise.
    """
    Z, Y, X = zyx_shape
    vol = np.zeros((Z, Y, X), dtype=np.uint8)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".roi"):
                continue
            data = zf.read(name)
            try:
                roi = parse_imagej_roi_bytes(data)
            except Exception as e:
                print(f"Skipping ROI {name} in {os.path.basename(zip_path)}: {e}")
                continue

            coords = roi.get("coords", None)
            slice_idx = roi.get("slice", None)
            if coords is None or len(coords) < 3 or slice_idx is None:
                continue

            z_idx = int(slice_idx) - 1  # ImageJ is 1-based
            if 0 <= z_idx < Z:
                rr, cc = sk_polygon(coords[:, 1], coords[:, 0], shape=(Y, X))
                rr = np.clip(rr, 0, Y - 1)
                cc = np.clip(cc, 0, X - 1)
                vol[z_idx, rr, cc] = 1

    # Fill holes for each used slice
    used = np.where(np.any(vol, axis=(1, 2)))[0]
    for z in used:
        vol[z] = binary_fill_holes(vol[z]).astype(np.uint8)

    return vol

# ---------------------------
# Tight crop (reduce file size) + compute offsets
# ---------------------------
def crop_volume_and_get_offset(vol):
    """
    Return cropped_vol, (off_x, off_y) in pixels, and z_used indices.
    Crops to the minimal bbox over all slices that have any foreground.
    """
    Z, Y, X = vol.shape
    z_used = np.where(np.any(vol, axis=(1, 2)))[0]
    if z_used.size == 0:
        return None, (0, 0), z_used

    ys, xs = np.where(np.any(vol[z_used, :, :], axis=0))
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    cropped = vol[:, y0:y1, x0:x1].copy()
    return cropped, (x0, y0), z_used

# ---------------------------
# Save one instance (cropped)
# ---------------------------
def save_instance(out_dir, idx, volume_cropped, offset_xy_px, vxy_um, vz_um, z_used, source_zip, source_czi):
    """
    Save cropped volume + meta like predicted outputs:
      - vol_XXXX.npy
      - vol_XXXX_meta.json
    Meta fields:
      voxel_size_*_um  -> microns
      offset_xy_global_px -> [x0, y0] top-left crop in original full frame
      z_min_um -> set so that Z-mid of the occupied slices maps to z=0
    """
    name = f"vol_{idx:04d}"
    np.save(os.path.join(out_dir, name + ".npy"), volume_cropped.astype(np.uint8), allow_pickle=False)

    Z, Yc, Xc = map(int, volume_cropped.shape)

    # Z-mid in slice indices (on the original Z axis)
    if z_used.size > 0:
        z_mid = 0.5 * (float(z_used[0]) + float(z_used[-1]))
    else:
        z_mid = (Z - 1) * 0.5  # fallback

    # z = z_index * vz_um + z_min_um ; enforce z(z_mid) = 0  =>  z_min_um = - z_mid * vz_um
    z_min_um = - z_mid * float(vz_um)
    z_max_um = z_min_um + Z * float(vz_um)
    z0_index = int(round(- z_min_um / float(vz_um))) if vz_um != 0 else int(round(z_mid))

    meta = {
        "instance_id": int(idx),
        "source_instance": os.path.splitext(os.path.basename(source_zip))[0],
        "volume_shape_zyx": [int(Z), int(Yc), int(Xc)],
        "voxel_size_xy_um": float(vxy_um),
        "voxel_size_z_um":  float(vz_um),
        "offset_xy_global_px": [int(offset_xy_px[0]), int(offset_xy_px[1])],
        "z_min_um": float(z_min_um),
        "z_max_um": float(z_max_um),
        "z0_index": int(z0_index),
        "source_image": os.path.basename(source_czi),
        "is_gt": True
    }
    with open(os.path.join(out_dir, name + "_meta.json"), "w") as jf:
        json.dump(meta, jf, indent=2)

# ---------------------------
# Main
# ---------------------------
def main():
    print("=== GT ROI ZIP -> per-instance CROPPED volumes (prediction-compatible, microns) ===")

    czi_path = select_file("Select CZI z-stack", [("CZI files", "*.czi"), ("All files", "*.*")])
    if not czi_path:
        print("No CZI selected. Exiting.")
        return

    roizip_dir = select_folder("Select folder containing ROI ZIP sets")
    if not roizip_dir:
        print("No ROI ZIP folder selected. Exiting.")
        return

    out_dir = select_folder("Select OUTPUT folder for GT volumes")
    if not out_dir:
        print("No output folder selected. Exiting.")
        return

    cleanup_output_dir(out_dir)
    print("Output folder cleaned:", out_dir)

    (Z, Y, X), vxy_um_raw, vz_um_raw = load_czi_shape_and_scaling(czi_path)

    # FINAL GUARDED MICRONS
    vxy_um = guard_um(vxy_um_raw) or 0.03225
    vz_um  = guard_um(vz_um_raw)  or vxy_um
    print(f"[USING microns] vxy_um={vxy_um:.8f}  vz_um={vz_um:.8f}")

    zip_files = sorted([f for f in os.listdir(roizip_dir) if f.lower().endswith(".zip")])
    if not zip_files:
        print("No ZIP files found in", roizip_dir)
        return

    count_saved = 0
    for i, zname in enumerate(zip_files):
        zip_path = os.path.join(roizip_dir, zname)
        vol_full = build_gt_volume_from_zip(zip_path, (Z, Y, X))
        if np.sum(vol_full) == 0:
            print("Empty volume for", zname, "- skipping.")
            continue

        vol_crop, (x0, y0), z_used = crop_volume_and_get_offset(vol_full)
        if vol_crop is None or vol_crop.size == 0:
            print("Invalid crop for", zname, "- skipping.")
            continue

        save_instance(out_dir, count_saved, vol_crop, (x0, y0),
                      vxy_um, vz_um, z_used, zip_path, czi_path)
        print(f"[OK] GT #{count_saved} from {zname} | crop=({y0}:{y0+vol_crop.shape[1]}, {x0}:{x0+vol_crop.shape[2]})  Z_used={z_used[0]}..{z_used[-1]}  shape={vol_crop.shape}")

        count_saved += 1

    print(f"Done. Saved {count_saved} GT volumes to {out_dir}")

if __name__ == "__main__":
    main()
