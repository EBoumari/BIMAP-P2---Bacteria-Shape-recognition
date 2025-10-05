# visualize_3d_with_bg_plane.py
# Show 3D volumes at their global XY positions and render the background image
# as a textured plane at Z = 0. Adds a metric XY grid (minor=2 µm, major=10 µm).
# Meshes are white; scene background is black. ASCII-only comments.

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

try:
    from aicsimageio import AICSImage
    HAS_AICS = True
except Exception:
    HAS_AICS = False


# ---------------------------
# UI helpers
# ---------------------------
def select_folder(title):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title=title)

def select_image_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select background image (will be placed at Z=0)",
        filetypes=[("Images", "*.czi *.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
    )


# ---------------------------
# Image loading
# ---------------------------
def load_image_rgb_any(path):
    """
    Load image and return an 8-bit RGB array (H, W, 3).
    Supports CZI via AICSImage if available, otherwise skimage.io for common formats.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".czi":
        if not HAS_AICS:
            raise RuntimeError("AICSImage not available. Install aicsimageio to read CZI.")
        img = AICSImage(path)
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
    else:
        arr = io.imread(path)
        if arr.ndim == 2:
            rgb = gray2rgb(arr)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            rgb = arr[:, :, :3]
        elif arr.ndim == 3 and arr.shape[2] == 3:
            rgb = arr
        else:
            first = arr[..., 0]
            rgb = gray2rgb(first)

    # convert to uint8 0..255 for texture
    if rgb.dtype != np.uint8:
        rgb = rescale_intensity(rgb, in_range="image", out_range=(0, 255)).astype(np.uint8)
    return rgb


# ---------------------------
# Volumes + meta loading
# ---------------------------
def load_volumes_with_meta(folder):
    """
    Load vol_*.npy and corresponding vol_*_meta.json.
    Returns lists: volumes, metas.
    """
    vols = []
    metas = []
    files = sorted([f for f in os.listdir(folder) if f.startswith("vol_") and f.endswith(".npy")])
    for f in files:
        base = os.path.splitext(f)[0]
        meta_path = os.path.join(folder, base + "_meta.json")
        vol_path = os.path.join(folder, f)
        if not os.path.exists(meta_path):
            print("Warning: missing meta for", f)
            continue
        try:
            vol = np.load(vol_path)
            with open(meta_path, "r") as jf:
                meta = json.load(jf)
            vols.append(vol)
            metas.append(meta)
        except Exception as e:
            print("Warning: could not load", f, "error:", e)
    print("Loaded {} volumes from {}".format(len(vols), folder))
    return vols, metas


# ---------------------------
# Mesh creation and placement
# ---------------------------
def mesh_from_volume_global(vol, meta):
    """
    Create a PyVista mesh from a binary volume and place it in global coordinates (microns).
    - marching_cubes spacing is (dz, dy, dx) in microns
    - convert vertices from (z,y,x) to (x,y,z)
    - apply XY translation by offset_xy_global_px * voxel_size_xy_um
    - apply Z translation by z_min_um so that slice 0 is at that z
    """
    if np.sum(vol) == 0:
        return None

    vxy = float(meta.get("voxel_size_xy_um", 0.03225))
    vz = float(meta.get("voxel_size_z_um", vxy))
    offset_xy = meta.get("offset_xy_global_px", [0, 0])
    z_min_um = float(meta.get("z_min_um", 0.0))

    try:
        verts, faces, _, _ = marching_cubes(vol.astype(np.float32), level=0.5, spacing=(vz, vxy, vxy))
    except Exception as e:
        print("marching_cubes failed:", e)
        return None

    verts_xyz = verts[:, [2, 1, 0]].copy()
    ox_um = float(offset_xy[0]) * vxy
    oy_um = float(offset_xy[1]) * vxy
    verts_xyz[:, 0] += ox_um
    verts_xyz[:, 1] += oy_um
    verts_xyz[:, 2] += z_min_um

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces]).ravel()
    mesh = pv.PolyData(verts_xyz, faces_pv)
    return mesh


# ---------------------------
# Background plane at Z = 0  (+ sizes)
# ---------------------------
def make_bg_plane_with_texture(bg_rgb, voxel_size_xy_um):
    """
    Create a PyVista plane at Z=0 sized to the image in microns and attach the image as texture.
    Also returns plane sizes in microns for grid drawing.
    """
    H, W = bg_rgb.shape[:2]
    size_x_um = W * float(voxel_size_xy_um)
    size_y_um = H * float(voxel_size_xy_um)

    center = (size_x_um * 0.5, size_y_um * 0.5, 0.0)
    plane = pv.Plane(center=center, direction=(0, 0, 1), i_size=size_x_um, j_size=size_y_um)

    # Flip vertically so texture matches image coordinates (top-left origin)
    bg_tex = np.flipud(bg_rgb.copy())
    texture = pv.Texture(bg_tex, falseorigin=True)

    return plane, texture, size_x_um, size_y_um


# ---------------------------
# XY metric grid (Z=0)
# ---------------------------
def _is_major(val_um, major_every_um):
    r = val_um % major_every_um
    return (r < 1e-6) or (major_every_um - r < 1e-6)

def add_xy_metric_grid(plotter, size_x_um, size_y_um, step_um=2.0, major_every_um=10.0, z=0.0):
    """
    Draw an XY grid at Z=z. Minor lines every step_um, major lines every major_every_um.
    """
    if step_um <= 0:
        return

    z += 1e-3  # offset to avoid z-fighting with the textured plane

    minor_color = "white"
    major_color = "white"
    minor_opacity = 0.55
    major_opacity = 0.95
    minor_width = 2
    major_width = 3

    nx = int(np.floor(size_x_um / step_um)) + 1
    ny = int(np.floor(size_y_um / step_um)) + 1

    # verticals (x constant)
    for i in range(nx + 1):
        x = i * step_um
        line = pv.Line((x, 0.0, z), (x, size_y_um, z))
        if _is_major(x, major_every_um):
            plotter.add_mesh(line, color=major_color, opacity=major_opacity,
                             line_width=major_width, name=f"grid_x_major_{i}", pickable=False)
        else:
            plotter.add_mesh(line, color=minor_color, opacity=minor_opacity,
                             line_width=minor_width, name=f"grid_x_minor_{i}", pickable=False)

    # horizontals (y constant)
    for j in range(ny + 1):
        y = j * step_um
        line = pv.Line((0.0, y, z), (size_x_um, y, z))
        if _is_major(y, major_every_um):
            plotter.add_mesh(line, color=major_color, opacity=major_opacity,
                             line_width=major_width, name=f"grid_y_major_{j}", pickable=False)
        else:
            plotter.add_mesh(line, color=minor_color, opacity=minor_opacity,
                             line_width=minor_width, name=f"grid_y_minor_{j}", pickable=False)


# ---------------------------
# Visualization
# ---------------------------
def main():
    print("=== Visualize 3D volumes with background plane at Z=0 + XY 2µm grid ===")

    vols_dir = select_folder("Select folder containing vol_*.npy and vol_*_meta.json")
    if not vols_dir:
        print("No folder selected. Exiting.")
        return

    bg_path = select_image_file()
    if not bg_path:
        print("No background image selected. Exiting.")
        return

    # Load data
    vols, metas = load_volumes_with_meta(vols_dir)
    if not vols:
        print("No volumes to show.")
        return

    # Use voxel size from the first volume metadata for the background scaling
    vxy = float(metas[0].get("voxel_size_xy_um", 0.03225))

    # Load background image and build plane at Z=0
    bg_rgb = load_image_rgb_any(bg_path)
    plane, texture, size_x_um, size_y_um = make_bg_plane_with_texture(bg_rgb, vxy)

    # Create plotter
    pv.global_theme.background = "black"
    pv.global_theme.smooth_shading = False
    plotter = pv.Plotter()
    plotter.set_background("black")

    # Add background plane with texture at Z=0
    plotter.add_mesh(plane, texture=texture, name="bg_plane", pickable=False)

    # Add XY metric grid at Z=0: minor=2 µm, major=10 µm
    add_xy_metric_grid(plotter, size_x_um, size_y_um, step_um=2.0, major_every_um=10.0, z=0.0)

    # Add each volume mesh in white at global position
    any_added = False
    for vol, meta in zip(vols, metas):
        mesh = mesh_from_volume_global(vol, meta)
        if mesh is None or mesh.n_points == 0:
            continue
        plotter.add_mesh(mesh, color="magenta", opacity=0.9, smooth_shading=False)
        any_added = True

    if not any_added:
        print("No valid meshes added. Exiting.")
        return

    # Axes (keep), but disable default show_grid() to avoid clutter with metric grid
    plotter.add_axes(line_width=2)
    # On-screen note about grid spacing
    plotter.add_text("XY grid: 2 microns", font_size=12, color="blue", position="upper_right")
    # Camera
    plotter.reset_camera()
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.2)

    print("Ready. BG at Z=0, XY grid: minor=2 µm, major=10 µm. Mesh color is white.")
    plotter.show()


if __name__ == "__main__":
    main()
