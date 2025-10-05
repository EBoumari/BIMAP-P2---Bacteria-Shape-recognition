# 3D_pred_gt_overlay_with_2um_grid.py
# Overlay predicted (white) + GT (magenta) volumes with optional background
# and draw a metric XY grid at Z=0 with 2-micron spacing.
# ASCII-only comments.

import os
import json
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pyvista as pv
from skimage import io
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from skimage.measure import marching_cubes

# ---------- UI ----------
def pick_folder(title):
    root = tk.Tk(); root.withdraw()
    return filedialog.askdirectory(title=title)

def pick_image(title):
    root = tk.Tk(); root.withdraw()
    return filedialog.askopenfilename(
        title=title,
        filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
    )

# ---------- IO helpers ----------
def load_vols_meta(folder):
    vols, metas = [], []
    if not folder or not os.path.isdir(folder):
        print(f"[WARN] folder not found: {folder}")
        return vols, metas
    npys = sorted([f for f in os.listdir(folder) if f.startswith("vol_") and f.endswith(".npy")])
    print(f"[LOAD] scanning {folder} -> {len(npys)} candidates")
    for f in npys:
        base = os.path.splitext(f)[0]
        mp = os.path.join(folder, base + "_meta.json")
        vp = os.path.join(folder, f)
        if not os.path.exists(mp):
            print(f"[SKIP] meta missing for {f}")
            continue
        try:
            vol = np.load(vp, mmap_mode="r")
            with open(mp, "r") as jf:
                meta = json.load(jf)
            vols.append(vol)
            metas.append(meta)
        except Exception as e:
            print(f"[SKIP] failed to load {f}: {e}")
    print(f"[OK] loaded {len(vols)} entries")
    return vols, metas

def load_bg_rgb(path):
    arr = io.imread(path)
    if arr.ndim == 2:
        rgb = gray2rgb(arr)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        rgb = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        rgb = arr
    else:
        rgb = gray2rgb(arr[..., 0])
    if rgb.dtype != np.uint8:
        rgb = rescale_intensity(rgb, in_range="image", out_range=(0, 255)).astype(np.uint8)
    return rgb

# ---------- Geometry ----------
def volume_to_mesh_global(vol, meta, who="UNK"):
    # build surface in microns and place in global coordinates
    voxels = int(np.sum(vol > 0))
    if voxels == 0:
        print(f"[{who}] empty volume -> skip")
        return None

    vxy = float(meta.get("voxel_size_xy_um", 0.03225))
    vz  = float(meta.get("voxel_size_z_um", vxy))
    off = meta.get("offset_xy_global_px", [0, 0])
    zmin = float(meta.get("z_min_um", 0.0))

    try:
        verts, faces, _, _ = marching_cubes(vol.astype(np.float32), level=0.5,
                                            spacing=(vz, vxy, vxy))
    except Exception as e:
        print(f"[{who}] marching_cubes failed: {e}")
        return None

    # swap to (x,y,z)
    verts_xyz = verts[:, [2, 1, 0]].copy()
    ox_um = float(off[0]) * vxy
    oy_um = float(off[1]) * vxy
    verts_xyz[:, 0] += ox_um
    verts_xyz[:, 1] += oy_um
    verts_xyz[:, 2] += zmin

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces]).ravel()
    return pv.PolyData(verts_xyz, faces_pv)

def make_bg_plane(bg_rgb, vxy_um, origin_px=(0, 0)):
    H, W = bg_rgb.shape[:2]
    size_x_um = W * float(vxy_um)
    size_y_um = H * float(vxy_um)
    cx = origin_px[0] * vxy_um + size_x_um * 0.5
    cy = origin_px[1] * vxy_um + size_y_um * 0.5
    plane = pv.Plane(center=(cx, cy, 0.0), direction=(0, 0, 1),
                     i_size=size_x_um, j_size=size_y_um)
    tex = pv.Texture(np.flipud(bg_rgb.copy()), falseorigin=True)
    return plane, tex, size_x_um, size_y_um

def compute_scene_xy_bounds(meshes):
    # find global min/max over X,Y from a list of meshes
    xs, ys = [], []
    for m in meshes:
        if m is None or m.n_points == 0: 
            continue
        pts = m.points
        xs.append((pts[:, 0].min(), pts[:, 0].max()))
        ys.append((pts[:, 1].min(), pts[:, 1].max()))
    if not xs or not ys:
        return (0.0, 0.0)
    x_min = min(a for a, b in xs); x_max = max(b for a, b in xs)
    y_min = min(a for a, b in ys); y_max = max(b for a, b in ys)
    return (max(0.0, x_max - x_min), max(0.0, y_max - y_min))

def add_xy_metric_grid(plotter, size_x_um, size_y_um, step_um=2.0, z=0.0,
                       color="gray", opacity=0.35, line_width=1):
    """
    Draw a metric grid on Z=0 every `step_um` microns across [0..size_x_um]x[0..size_y_um].
    """
    if step_um <= 0:
        return
    nx = int(np.floor(size_x_um / step_um)) + 1
    ny = int(np.floor(size_y_um / step_um)) + 1

    # vertical lines (x = k*step)
    for i in range(nx + 1):
        x = i * step_um
        a = (x, 0.0, z)
        b = (x, size_y_um, z)
        line = pv.Line(a, b)
        plotter.add_mesh(line, color=color, opacity=opacity, line_width=line_width, name=f"grid_x_{i}", pickable=False)

    # horizontal lines (y = k*step)
    for j in range(ny + 1):
        y = j * step_um
        a = (0.0, y, z)
        b = (size_x_um, y, z)
        line = pv.Line(a, b)
        plotter.add_mesh(line, color=color, opacity=opacity, line_width=line_width, name=f"grid_y_{j}", pickable=False)

# ---------- Main ----------
def main():
    print("=== Overlay Pred (white) + GT (magenta) + BG (Z=0) + 2um XY grid ===")

    pred_dir = pick_folder("Select PRED volumes folder (vol_*.npy + meta)")
    gt_dir   = pick_folder("Select GT volumes folder (vol_*.npy + meta)")
    bg_path  = pick_image("Select background image (optional; Esc to skip)")

    pred_vols, pred_metas = load_vols_meta(pred_dir)
    gt_vols,   gt_metas   = load_vols_meta(gt_dir)

    # choose vxy (for background plane scale)
    vxy_vals = []
    if pred_metas: vxy_vals.append(float(pred_metas[0].get("voxel_size_xy_um", 0.03225)))
    if gt_metas:   vxy_vals.append(float(gt_metas[0].get("voxel_size_xy_um", 0.03225)))
    vxy_for_bg = vxy_vals[0] if vxy_vals else 0.03225

    pv.global_theme.background = "black"
    pv.global_theme.smooth_shading = False
    plot = pv.Plotter()
    plot.set_background("black")

    # Make meshes first (so we can compute bounds if no BG)
    pred_meshes, gt_meshes = [], []

    for i, (vol, meta) in enumerate(zip(pred_vols, pred_metas)):
        m = volume_to_mesh_global(vol, meta, who=f"PRED vol_{i:04d}")
        if m is not None and m.n_points > 0:
            plot.add_mesh(m, color="white", opacity=0.85, smooth_shading=False, name=f"pred_{i:04d}")
            pred_meshes.append(m)

    for i, (vol, meta) in enumerate(zip(gt_vols, gt_metas)):
        m = volume_to_mesh_global(vol, meta, who=f"GT   vol_{i:04d}")
        if m is not None and m.n_points > 0:
            plot.add_mesh(m, color=(1.0, 0.2, 1.0), opacity=0.65, smooth_shading=False, name=f"gt_{i:04d}")
            gt_meshes.append(m)

    print(f"[PRED] meshes added: {len(pred_meshes)}")
    print(f"[GT]   meshes added: {len(gt_meshes)}")

    # Background plane + grid extent
    size_x_um = size_y_um = None
    if bg_path:
        try:
            bg_rgb = load_bg_rgb(bg_path)
            plane, tex, size_x_um, size_y_um = make_bg_plane(bg_rgb, vxy_for_bg, origin_px=(0, 0))
            plot.add_mesh(plane, texture=tex, name="bg_plane", pickable=False)
            print(f"[BG] plane px=({bg_rgb.shape[1]},{bg_rgb.shape[0]}) um=({size_x_um:.2f},{size_y_um:.2f})")
        except Exception as e:
            print(f"[BG] failed to add plane: {e}")

    # If no BG, estimate grid extent from scene bounds (pads to next 2 µm)
    if size_x_um is None or size_y_um is None:
        sx, sy = compute_scene_xy_bounds(pred_meshes + gt_meshes)
        # pad a little and snap up to multiples of 2µm
        sx = max(sx, 1.0); sy = max(sy, 1.0)
        step = 2.0
        size_x_um = step * np.ceil((sx + step) / step)
        size_y_um = step * np.ceil((sy + step) / step)
        print(f"[GRID] no BG -> extent estimated um=({size_x_um:.2f},{size_y_um:.2f})")

    # Add 2 µm XY grid at Z=0
    add_xy_metric_grid(plot, size_x_um, size_y_um, step_um=2.0, z=0.0,
                       color="cyan", opacity=0.6, line_width=1)
    print("[GRID] 2 µm grid added on Z=0")

    # Depth peeling (if available)
    try:
        plot.enable_depth_peeling(number_of_peels=200, occlusion_ratio=0.0)
        print("[RENDER] depth peeling enabled")
    except Exception as e:
        print(f"[RENDER] depth peeling unavailable: {e}")

    # Axes/grid text + camera
    plot.add_axes(line_width=2)
    plot.add_text("Legend: white = Predicted, magenta = Ground Truth | XY grid: 2 microns",
                  font_size=12, color="blue", position="upper_right")
    plot.show_bounds(grid='front', color="gray", xlabel="X (µm)", ylabel="Y (µm)", zlabel="Z (µm)")
    plot.reset_camera()
    plot.camera_position = "iso"
    plot.camera.zoom(1.2)

    if len(pred_meshes) == 0 and len(gt_meshes) == 0:
        print("[WARN] No meshes to display.")

    print("Ready. White = Pred, Magenta = GT, BG at Z=0. XY grid: 2 µm.")
    plot.show()

if __name__ == "__main__":
    main()
