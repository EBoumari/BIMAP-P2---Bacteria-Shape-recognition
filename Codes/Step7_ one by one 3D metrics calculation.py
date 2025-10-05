# evaluate_and_visualize.py
# One-stop: load pred/gt volumes, compute IoU/Dice (batch), save CSV/JSON,
# visualize with labels, then interactively compute IoU/Dice for a chosen pair.
# ASCII-only comments/prints for Windows console safety.

import os, sys, json, csv, math, argparse
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import io
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from skimage.measure import marching_cubes

import pyvista as pv

# ---------------------------
# UI helpers
# ---------------------------
def ask_folder(title):
    root = tk.Tk(); root.withdraw()
    return filedialog.askdirectory(title=title)

def ask_image():
    root = tk.Tk(); root.withdraw()
    return filedialog.askopenfilename(
        title="Select background image (optional)",
        filetypes=[("Images", "*.czi *.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
    )

# ---------------------------
# Image loading (no CZI fallback)
# ---------------------------
def load_image_rgb(path):
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".czi":
        try:
            from aicsimageio import AICSImage
        except Exception:
            print("[BG] aicsimageio not installed; cannot read CZI. Skipping background.")
            return None
        try:
            img = AICSImage(path)
            yx = img.get_image_data("YX", S=0, T=0, C=0)
            rgb = gray2rgb(yx)
        except Exception as e:
            print("[BG] failed to read CZI:", e)
            return None
    else:
        try:
            arr = io.imread(path)
        except Exception as e:
            print("[BG] failed to read image:", e); return None
        if arr.ndim == 2:
            rgb = gray2rgb(arr)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[:, :, :3]
        else:
            rgb = gray2rgb(arr[..., 0])

    if rgb.dtype != np.uint8:
        rgb = rescale_intensity(rgb, in_range="image", out_range=(0, 255)).astype(np.uint8)
    return rgb

def make_bg_plane(rgb, vxy_um):
    if rgb is None: return None, None
    H, W = rgb.shape[:2]
    sx = W * float(vxy_um)
    sy = H * float(vxy_um)
    center = (sx * 0.5, sy * 0.5, 0.0)
    plane = pv.Plane(center=center, direction=(0, 0, 1), i_size=sx, j_size=sy)
    # flip vertically so texture matches image Y-down
    tex = pv.Texture(np.flipud(rgb.copy()), falseorigin=True)
    return plane, tex

# ---------------------------
# I/O helpers for vols + meta
# ---------------------------
def load_vols_meta(folder):
    vols, metas, names = [], [], []
    if not folder or not os.path.isdir(folder):
        print("[ERR] invalid folder:", folder); return vols, metas, names
    files = sorted([f for f in os.listdir(folder) if f.startswith("vol_") and f.endswith(".npy")])
    print(f"[LOAD] {len(files)} volumes in {folder}")
    for f in files:
        base = os.path.splitext(f)[0]
        meta_path = os.path.join(folder, base + "_meta.json")
        vol_path = os.path.join(folder, f)
        if not os.path.exists(meta_path):
            print("[WARN] missing meta for", f); continue
        try:
            vol = np.load(vol_path)
            with open(meta_path, "r") as jf:
                meta = json.load(jf)
            vols.append(vol); metas.append(meta); names.append(base)
        except Exception as e:
            print("[WARN] could not load", f, ":", e)
    print(f"[LOAD] loaded {len(vols)} pairs")
    return vols, metas, names

# ---------------------------
# Geometry utils
# ---------------------------
def world_bounds_um(vol, meta):
    vxy = float(meta.get("voxel_size_xy_um", 0.03225))
    vz  = float(meta.get("voxel_size_z_um",  vxy))
    off = meta.get("offset_xy_global_px", [0, 0])
    z0  = float(meta.get("z_min_um", 0.0))
    nz = np.argwhere(vol > 0)
    if nz.size == 0: return None
    zs, ys, xs = nz[:,0], nz[:,1], nz[:,2]
    xw = xs * vxy + float(off[0]) * vxy
    yw = ys * vxy + float(off[1]) * vxy
    zw = zs * vz  + z0
    return (xw.min(), xw.max(), yw.min(), yw.max(), zw.min(), zw.max())

def aabb_overlap_xy(b1, b2, margin_um=0.0):
    if b1 is None or b2 is None: return False
    x1min, x1max, y1min, y1max, _, _ = b1
    x2min, x2max, y2min, y2max, _, _ = b2
    return not (x1max < x2min - margin_um or x2max < x1min - margin_um or
                y1max < y2min - margin_um or y2max < y1min - margin_um)

def mesh_from_volume_global(vol, meta):
    if np.sum(vol) == 0: return None
    vxy = float(meta.get("voxel_size_xy_um", 0.03225))
    vz  = float(meta.get("voxel_size_z_um",  vxy))
    off = meta.get("offset_xy_global_px", [0, 0])
    z0  = float(meta.get("z_min_um", 0.0))
    try:
        verts, faces, _, _ = marching_cubes(vol.astype(np.float32), level=0.5, spacing=(vz, vxy, vxy))
    except Exception as e:
        print("marching_cubes failed:", e); return None
    verts_xyz = verts[:, [2, 1, 0]].copy()
    verts_xyz[:, 0] += float(off[0]) * vxy
    verts_xyz[:, 1] += float(off[1]) * vxy
    verts_xyz[:, 2] += z0
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces]).ravel()
    return pv.PolyData(verts_xyz, faces_pv)

# ---------------------------
# IoU/Dice (pair) on shared grid in world microns
# ---------------------------
def iou_dice_pair(volA, metaA, volB, metaB):
    vxyA = float(metaA.get("voxel_size_xy_um", 0.03225))
    vzA  = float(metaA.get("voxel_size_z_um",  vxyA))
    offA = metaA.get("offset_xy_global_px", [0, 0])
    z0A  = float(metaA.get("z_min_um", 0.0))

    vxyB = float(metaB.get("voxel_size_xy_um", 0.03225))
    vzB  = float(metaB.get("voxel_size_z_um",  vxyB))
    offB = metaB.get("offset_xy_global_px", [0, 0])
    z0B  = float(metaB.get("z_min_um", 0.0))

    bA = world_bounds_um(volA, metaA)
    bB = world_bounds_um(volB, metaB)
    if bA is None or bB is None:
        return 0.0, 0.0

    xmin = min(bA[0], bB[0]); xmax = max(bA[1], bB[1])
    ymin = min(bA[2], bB[2]); ymax = max(bA[3], bB[3])
    zmin = min(bA[4], bB[4]); zmax = max(bA[5], bB[5])

    vxy = max(vxyA, vxyB)
    vz  = max(vzA,  vzB)

    nx = max(1, int(math.floor((xmax - xmin)/vxy)) + 1)
    ny = max(1, int(math.floor((ymax - ymin)/vxy)) + 1)
    nz = max(1, int(math.floor((zmax - zmin)/vz )) + 1)

    A_local = np.zeros((nz, ny, nx), dtype=np.uint8)
    B_local = np.zeros((nz, ny, nx), dtype=np.uint8)

    idxA = np.argwhere(volA > 0)
    if idxA.size > 0:
        zA, yA, xA = idxA[:,0], idxA[:,1], idxA[:,2]
        xwA = xA * vxyA + float(offA[0]) * vxyA
        ywA = yA * vxyA + float(offA[1]) * vxyA
        zwA = zA * vzA  + z0A
        lx = np.floor((xwA - xmin)/vxy).astype(int)
        ly = np.floor((ywA - ymin)/vxy).astype(int)
        lz = np.floor((zwA - zmin)/vz ).astype(int)
        mask = (lx>=0)&(lx<nx)&(ly>=0)&(ly<ny)&(lz>=0)&(lz<nz)
        A_local[lz[mask], ly[mask], lx[mask]] = 1

    idxB = np.argwhere(volB > 0)
    if idxB.size > 0:
        zB, yB, xB = idxB[:,0], idxB[:,1], idxB[:,2]
        xwB = xB * vxyB + float(offB[0]) * vxyB
        ywB = yB * vxyB + float(offB[1]) * vxyB
        zwB = zB * vzB  + z0B
        lx = np.floor((xwB - xmin)/vxy).astype(int)
        ly = np.floor((ywB - ymin)/vxy).astype(int)
        lz = np.floor((zwB - zmin)/vz ).astype(int)
        mask = (lx>=0)&(lx<nx)&(ly>=0)&(ly<ny)&(lz>=0)&(lz<nz)
        B_local[lz[mask], ly[mask], lx[mask]] = 1

    inter = int(np.count_nonzero(A_local & B_local))
    voxA  = int(np.count_nonzero(A_local))
    voxB  = int(np.count_nonzero(B_local))
    union = voxA + voxB - inter
    iou  = (inter/union) if union>0 else 0.0
    dice = (2*inter/(voxA+voxB)) if (voxA+voxB)>0 else 0.0
    return float(iou), float(dice)

# ---------------------------
# Build IoU/Dice matrices + save reports
# ---------------------------
def build_and_save_reports(pred_dir, gt_dir, out_dir, iou_thr=0.10):
    os.makedirs(out_dir, exist_ok=True)
    pred_vols, pred_meta, pred_names = load_vols_meta(pred_dir)
    gt_vols,   gt_meta,   gt_names   = load_vols_meta(gt_dir)
    if not pred_vols or not gt_vols:
        print("[ERR] not enough data to evaluate"); return pred_vols, pred_meta, pred_names, gt_vols, gt_meta, gt_names

    nP, nG = len(pred_vols), len(gt_vols)
    bP = [world_bounds_um(v,m) for v,m in zip(pred_vols, pred_meta)]
    bG = [world_bounds_um(v,m) for v,m in zip(gt_vols,   gt_meta)]

    iou = np.zeros((nP,nG), dtype=np.float32)
    dice= np.zeros((nP,nG), dtype=np.float32)
    pairs = []

    for i in range(nP):
        for j in range(nG):
            if not aabb_overlap_xy(bP[i], bG[j], margin_um=0.2):
                continue
            ij_iou, ij_dice = iou_dice_pair(pred_vols[i], pred_meta[i], gt_vols[j], gt_meta[j])
            iou[i,j]  = ij_iou
            dice[i,j] = ij_dice
            pairs.append((pred_names[i], gt_names[j], ij_iou, ij_dice))

    # Save pair_stats.csv
    pair_csv = os.path.join(out_dir, "pair_stats.csv")
    with open(pair_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pred","gt","iou","dice"])
        for pn, gn, i, d in pairs:
            w.writerow([pn, gn, f"{i:.6f}", f"{d:.6f}"])
    print("[OUT]", pair_csv)

    # Hungarian on (1 - IoU)
    cost = 1.0 - iou
    nP, nG = iou.shape
    if nP > nG:
        cost = np.hstack([cost, np.ones((nP, nP-nG), dtype=np.float32)])
    elif nG > nP:
        cost = np.vstack([cost, np.ones((nG-nP, nG), dtype=np.float32)])
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for r,c in zip(row_ind, col_ind):
        if r < len(pred_names) and c < len(gt_names):
            matches.append((r, c, float(iou[r,c]), float(dice[r,c])))

    TPs = [(r,c,i,d) for (r,c,i,d) in matches if i >= iou_thr]
    matched_pred = {r for (r,_,_,_) in TPs}
    matched_gt   = {c for (_,c,_,_) in TPs}
    FPs = [i for i in range(len(pred_names)) if i not in matched_pred]
    FNs = [j for j in range(len(gt_names))   if j not in matched_gt]

    mean_iou  = float(np.mean([m[2] for m in TPs])) if TPs else 0.0
    mean_dice = float(np.mean([m[3] for m in TPs])) if TPs else 0.0
    prec = len(TPs) / max(1, (len(TPs)+len(FPs)))
    rec  = len(TPs) / max(1, (len(TPs)+len(FNs)))
    f1   = (2*prec*rec)/max(1e-8, (prec+rec)) if (prec+rec)>0 else 0.0

    # Save matches.csv
    match_csv = os.path.join(out_dir, "matches.csv")
    with open(match_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pred","gt","iou","dice","label"])
        for (r,c,i,d) in TPs:
            w.writerow([pred_names[r], gt_names[c], f"{i:.6f}", f"{d:.6f}", "TP"])
        for i in FPs:
            w.writerow([pred_names[i], "", "0.000000", "0.000000", "FP"])
        for j in FNs:
            w.writerow(["", gt_names[j], "0.000000", "0.000000", "FN"])
    print("[OUT]", match_csv)

    # Save summary.json
    summary = {
        "pred_count": len(pred_names), "gt_count": len(gt_names),
        "TP": len(TPs), "FP": len(FPs), "FN": len(FNs),
        "iou_threshold": iou_thr,
        "mean_iou": round(mean_iou, 6),
        "mean_dice": round(mean_dice, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[OUT]", os.path.join(out_dir, "summary.json"))

    return pred_vols, pred_meta, pred_names, gt_vols, gt_meta, gt_names

# ---------------------------
# Visualization with labels
# ---------------------------
def visualize_with_labels(pred_vols, pred_meta, pred_names, gt_vols, gt_meta, gt_names, bg_rgb):
    # choose background voxel size from any meta if available
    vxy = None
    if pred_meta: vxy = float(pred_meta[0].get("voxel_size_xy_um", 0.03225))
    if vxy is None and gt_meta: vxy = float(gt_meta[0].get("voxel_size_xy_um", 0.03225))
    if vxy is None: vxy = 0.03225

    plane, tex = make_bg_plane(bg_rgb, vxy) if bg_rgb is not None else (None, None)

    plotter = pv.Plotter()
    plotter.set_background("black")

    if plane is not None and tex is not None:
        plotter.add_mesh(plane, texture=tex, name="bg_plane")

    # add GT meshes (magenta) with labels "G####"
    for vol, meta, name in zip(gt_vols, gt_meta, gt_names):
        m = mesh_from_volume_global(vol, meta)
        if m is None or m.n_points == 0: continue
        plotter.add_mesh(m, color=(1.0, 0.0, 1.0), opacity=0.7, smooth_shading=False)
        # label at mesh center
        plotter.add_point_labels([m.center], [name.replace("vol_", "G")], point_size=0, font_size=16, text_color="magenta")

    # add Pred meshes (white) with labels "P####"
    for vol, meta, name in zip(pred_vols, pred_meta, pred_names):
        m = mesh_from_volume_global(vol, meta)
        if m is None or m.n_points == 0: continue
        plotter.add_mesh(m, color="white", opacity=0.9, smooth_shading=False)
        plotter.add_point_labels([m.center], [name.replace("vol_", "P")], point_size=0, font_size=16, text_color="white")

    plotter.add_axes(line_width=2)
    plotter.show_grid(color="gray")
    try:
        plotter.enable_depth_peeling()
    except Exception:
        pass

    print("[VIEW] Close the window to continue...")
    plotter.show()

# ---------------------------
# Interactive pair query (after visualization)
# ---------------------------
def interactive_pair_queries(pred_vols, pred_meta, pred_names, gt_vols, gt_meta, gt_names):
    if not pred_vols or not gt_vols:
        print("[INFO] no data for interactive queries"); return
    # build simple dicts
    pred_map = { name.replace("vol_", "P"): i for i, name in enumerate(pred_names) }
    gt_map   = { name.replace("vol_", "G"): j for j, name in enumerate(gt_names) }

    while True:
        root = tk.Tk(); root.withdraw()
        gt_label  = simpledialog.askstring("GT label",  "Enter GT label like G0007 (Cancel to stop):")
        if not gt_label: break
        pred_label= simpledialog.askstring("Pred label","Enter Pred label like P0083 (Cancel to stop):")
        if not pred_label: break

        if gt_label not in gt_map or pred_label not in pred_map:
            messagebox.showerror("Invalid label", "Label not found. Check the labels shown in the viewer.")
            continue

        gi = gt_map[gt_label]; pi = pred_map[pred_label]
        iou, dice = iou_dice_pair(pred_vols[pi], pred_meta[pi], gt_vols[gi], gt_meta[gi])
        msg = f"{pred_label} vs {gt_label}\nIoU = {iou:.6f}\nDice = {dice:.6f}"
        print("[PAIR]", msg.replace("\n", " | "))
        messagebox.showinfo("Pair IoU/Dice", msg)

# ---------------------------
# Main
# ---------------------------
def main():
    print("=== Evaluate, visualize with labels, and interactively query IoU/Dice ===")

    # Ask for folders and output path (explicitly ask for CSV/JSON location)
    pred_dir = ask_folder("Select PRED folder (vol_*.npy + _meta.json)")
    if not pred_dir: print("[ERR] no PRED folder"); return
    gt_dir   = ask_folder("Select GT folder (vol_*.npy + _meta.json)")
    if not gt_dir: print("[ERR] no GT folder"); return
    out_dir  = ask_folder("Select OUTPUT folder for CSV/JSON")
    if not out_dir: out_dir = pred_dir

    # Optional background
    bg_path = ask_image()
    bg_rgb = load_image_rgb(bg_path) if bg_path else None
    if bg_rgb is None:
        print("[BG] no background image or failed to load; continuing without BG.")

    # Batch metrics + save reports
    pred_vols, pred_meta, pred_names, gt_vols, gt_meta, gt_names = build_and_save_reports(
        pred_dir=pred_dir, gt_dir=gt_dir, out_dir=out_dir, iou_thr=0.10
    )

    # Visualization with labels
    visualize_with_labels(pred_vols, pred_meta, pred_names, gt_vols, gt_meta, gt_names, bg_rgb)

    # Interactive pair IoU/Dice loop
    interactive_pair_queries(pred_vols, pred_meta, pred_names, gt_vols, gt_meta, gt_names)

    print("Done.")

if __name__ == "__main__":
    main()
