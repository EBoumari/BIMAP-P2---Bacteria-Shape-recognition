# instances_to_3d_autothick_v3.py
# Solid 3D volume by rotating ALL interior mask pixels around the in-plane PCA axis.
# PCA axis is taken from interior points and chosen by max variance (major axis).
# All code and comments are ASCII only.

import os
import json
import math
import shutil
import tkinter as tk
from tkinter import filedialog

import numpy as np
from skimage.measure import find_contours, approximate_polygon, marching_cubes
from scipy.ndimage import gaussian_filter1d, binary_fill_holes
from sklearn.decomposition import PCA


# ---------------------------
# UI
# ---------------------------
def select_folder(title):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title=title)

def cleanup_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Math
# ---------------------------
def rotation_matrix(axis, theta):
    """Rodrigues rotation matrix for a unit axis."""
    ux, uy, uz = axis
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
    ], dtype=float)


# ---------------------------
# Helpers
# ---------------------------
def interior_points_global_from_crop_mask(crop_mask, bbox_xywh):
    """Return all interior pixel centers as global integer (x,y) coordinates."""
    x0, y0, w, h = bbox_xywh
    yy, xx = np.nonzero(crop_mask > 0)
    xg = (xx + x0).astype(np.int32)
    yg = (yy + y0).astype(np.int32)
    return np.column_stack([xg, yg])  # (N,2)


# ---------------------------
# Core: rotate interior pixels around PCA axis
# ---------------------------
def build_solid_volume_by_rotating_pixels(
    crop_mask, bbox_xywh,
    voxel_size_xy_um=0.03225,
    voxel_size_z_um=None,
    num_steps=90,
    pad_xy_px=4,
    rotate_about="major"  # "major" or "minor"
):
    """
    Create a solid 3D volume by rotating ALL interior mask pixels around a PCA axis in the image plane.
    rotate_about:
      - "major": rotate around the principal (max variance) axis (desired)
      - "minor": rotate around the secondary axis
    """
    if voxel_size_z_um is None:
        voxel_size_z_um = voxel_size_xy_um

    # Interior points (global XY) for robust PCA
    pts_xy = interior_points_global_from_crop_mask(crop_mask, bbox_xywh)  # (N,2)
    if pts_xy.shape[0] < 3:
        return None

    # PCA on interior points (not just contour)
    pca2 = PCA(n_components=2)
    pca2.fit(pts_xy.astype(float))
    idx_major = int(np.argmax(pca2.explained_variance_))
    idx_minor = 1 - idx_major
    pc_major = pca2.components_[idx_major]
    pc_minor = pca2.components_[idx_minor]

    if rotate_about.lower() == "major":
        axis_2d = pc_major
    elif rotate_about.lower() == "minor":
        axis_2d = pc_minor
    else:
        axis_2d = pc_major  # default

    axis = np.array([axis_2d[0], axis_2d[1], 0.0], dtype=float)
    axis /= (np.linalg.norm(axis) + 1e-12)

    # Center at mean of interior points
    center_xy = pca2.mean_
    center3 = np.array([center_xy[0], center_xy[1], 0.0], dtype=float)

    # Build 3D points list from interior pixels (z=0 initially)
    pts3 = np.column_stack([pts_xy[:, 0].astype(float),
                            pts_xy[:, 1].astype(float),
                            np.zeros(pts_xy.shape[0])])

    # Sweep all points around the axis
    angles = np.linspace(0.0, 2.0 * np.pi, num_steps, endpoint=False)

    # First pass to get bounds
    all_xy = []
    all_z_pix = []
    for theta in angles:
        R = rotation_matrix(axis, theta)
        rot = (pts3 - center3) @ R.T + center3
        all_xy.append(rot[:, :2])
        all_z_pix.append(rot[:, 2])

    all_xy = np.vstack(all_xy)
    all_z_pix = np.concatenate(all_z_pix, axis=0)

    x_min = float(np.min(all_xy[:, 0])) - pad_xy_px
    x_max = float(np.max(all_xy[:, 0])) + pad_xy_px
    y_min = float(np.min(all_xy[:, 1])) - pad_xy_px
    y_max = float(np.max(all_xy[:, 1])) + pad_xy_px

    width = int(np.ceil(x_max - x_min))
    height = int(np.ceil(y_max - y_min))

    # Z bounds (in microns) come from geometry
    z_min_um = float(np.min(all_z_pix)) * float(voxel_size_xy_um)
    z_max_um = float(np.max(all_z_pix)) * float(voxel_size_xy_um)
    if abs(z_max_um - z_min_um) < 1e-6:
        z_min_um -= float(voxel_size_z_um)
        z_max_um += float(voxel_size_z_um)

    depth = int(np.ceil((z_max_um - z_min_um) / float(voxel_size_z_um))) + 1

    vol = np.zeros((depth, height, width), dtype=np.uint8)

    # vectorized z to index (handles arrays)
    def z_to_index(z_um):
        return np.round((np.asarray(z_um) - z_min_um) / float(voxel_size_z_um)).astype(int)

    # Second pass: stamp voxels
    x0, y0 = x_min, y_min
    for theta in angles:
        R = rotation_matrix(axis, theta)
        rot = (pts3 - center3) @ R.T + center3

        xi = np.round(rot[:, 0] - x0).astype(int)
        yi = np.round(rot[:, 1] - y0).astype(int)
        zi_um = rot[:, 2] * float(voxel_size_xy_um)
        zi = z_to_index(zi_um)

        m = (zi >= 0) & (zi < depth) & (yi >= 0) & (yi < height) & (xi >= 0) & (xi < width)
        vol[zi[m], yi[m], xi[m]] = 1

    # Fill holes slice-wise to get solid sections
    for z in range(depth):
        if np.any(vol[z]):
            vol[z] = binary_fill_holes(vol[z]).astype(np.uint8)

    z0_index = int(np.round((0.0 - z_min_um) / float(voxel_size_z_um)))
    offset_xy_global_px = (int(np.round(x_min)), int(np.round(y_min)))

    return vol, offset_xy_global_px, float(z_min_um), float(z_max_um), int(z0_index), {
        "explained_variance": pca2.explained_variance_.tolist(),
        "axis_used": "major" if rotate_about.lower() == "major" else "minor"
    }


# ---------------------------
# PLY writer
# ---------------------------
def save_ply_ascii(verts_xyz, faces, ply_path):
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex {}\n".format(verts_xyz.shape[0]))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("element face {}\n".format(faces.shape[0]))
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts_xyz:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(v[0], v[1], v[2]))
        for tri in faces:
            f.write("3 {} {} {}\n".format(int(tri[0]), int(tri[1]), int(tri[2])))


# ---------------------------
# Main
# ---------------------------
def main():
    print("=== Instances to solid 3D volumes (rotate pixels around PCA major axis) ===")

    inst_dir = select_folder("Select folder containing inst_*.npy and inst_*.json")
    if not inst_dir:
        print("No instances folder selected. Exiting.")
        return

    out_dir = select_folder("Select output folder for 3D volumes and meshes")
    if not out_dir:
        print("No output folder selected. Exiting.")
        return

    cleanup_output_dir(out_dir)
    print("Output folder cleaned:", out_dir)

    voxel_size_xy_um = 0.03225
    voxel_size_z_um = 0.03225
    num_steps = 90
    pad_xy_px = 4

    files = sorted([f for f in os.listdir(inst_dir) if f.startswith("inst_") and f.endswith(".npy")])
    count = 0
    for f in files:
        base = os.path.splitext(f)[0]
        mask_path = os.path.join(inst_dir, f)
        meta_path = os.path.join(inst_dir, base + "_meta.json")
        if not os.path.exists(meta_path):
            print("Warning: missing meta for", f)
            continue

        crop_mask = np.load(mask_path).astype(np.uint8)
        with open(meta_path, "r") as jf:
            meta = json.load(jf)
        bbox = meta["bbox"]  # [x, y, w, h]

        res = build_solid_volume_by_rotating_pixels(
            crop_mask, bbox,
            voxel_size_xy_um=voxel_size_xy_um,
            voxel_size_z_um=voxel_size_z_um,
            num_steps=num_steps,
            pad_xy_px=pad_xy_px,
            rotate_about="major"
        )
        if res is None:
            print("Warning: volume build failed for", f)
            continue

        volume, offset_xy, z_min_um, z_max_um, z0_index, pca_info = res
        if np.sum(volume) == 0:
            print("Warning: empty volume for", f)
            continue

        name = "vol_{:04d}".format(count)
        np.save(os.path.join(out_dir, name + ".npy"), volume)

        meta_out = {
            "instance_id": meta.get("instance_id", count),
            "source_instance": base,
            "volume_shape_zyx": [int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2])],
            "voxel_size_xy_um": float(voxel_size_xy_um),
            "voxel_size_z_um": float(voxel_size_z_um),
            "offset_xy_global_px": [int(offset_xy[0]), int(offset_xy[1])],
            "z_min_um": float(z_min_um),
            "z_max_um": float(z_max_um),
            "z0_index": int(z0_index),
            "pca_axis_used": pca_info["axis_used"],
            "pca_explained_variance": pca_info["explained_variance"],
            "source_image": meta.get("source_image", None)
        }
        with open(os.path.join(out_dir, name + "_meta.json"), "w") as jf:
            json.dump(meta_out, jf, indent=2)

        # Optional mesh export
        try:
            verts, faces, _, _ = marching_cubes(
                volume.astype(np.float32), level=0.5,
                spacing=(voxel_size_z_um, voxel_size_xy_um, voxel_size_xy_um)
            )
            verts_xyz = verts[:, [2, 1, 0]].copy()
            ox_um = offset_xy[0] * voxel_size_xy_um
            oy_um = offset_xy[1] * voxel_size_xy_um
            verts_xyz[:, 0] += ox_um
            verts_xyz[:, 1] += oy_um
            verts_xyz[:, 2] += z_min_um

            save_ply_ascii(verts_xyz, faces.astype(np.int32), os.path.join(out_dir, name + ".ply"))
        except Exception as e:
            print("Warning: mesh creation failed for", f, "error:", e)

        print("Saved solid 3D volume and mesh for", base, "| axis:", meta_out["pca_axis_used"])
        count += 1

    print("Done. Created", count, "solid 3D volumes in", out_dir)


if __name__ == "__main__":
    main()
