# compute_iou_dice.py
# Evaluate Pred vs GT volumes (vol_XXXX.npy + vol_XXXX_meta.json):
# - CLI args or GUI dialogs to get folders
# - Computes pairwise IoU/Dice on a shared local grid in WORLD microns
# - Hungarian matching
# - ALWAYS saves outputs: matches.csv, pair_stats.csv, summary.json

import os, json, csv, math, argparse, sys
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------
# UI helpers
# ---------------------------
def select_folder(title):
    root = tk.Tk(); root.withdraw()
    return filedialog.askdirectory(title=title)

# ---------------------------
# Data loading
# ---------------------------
def load_vols_with_meta(folder):
    vols, metas, names = [], [], []
    files = sorted([f for f in os.listdir(folder) if f.startswith("vol_") and f.endswith(".npy")])
    print(f"[LOAD] {len(files)} volumes in {folder}")
    for f in files:
        base = os.path.splitext(f)[0]
        meta_path = os.path.join(folder, base + "_meta.json")
        vol_path  = os.path.join(folder, f)
        if not os.path.exists(meta_path):
            print(f"[WARN] missing meta for {f}, skipping")
            continue
        try:
            vol = np.load(vol_path)
            with open(meta_path, "r") as jf:
                meta = json.load(jf)
            vols.append(vol)
            metas.append(meta)
            names.append(base)
        except Exception as e:
            print(f"[WARN] could not load {f}: {e}")
    print(f"[LOAD] loaded {len(vols)} vol+meta pairs")
    return vols, metas, names

# ---------------------------
# Bounds + overlap
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

# ---------------------------
# Pair IoU/Dice in shared local grid
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
        return 0.0, 0.0, {"voxA":0,"voxB":0,"voxInt":0,"voxUnion":0}

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
    return float(iou), float(dice), {"voxA":voxA,"voxB":voxB,"voxInt":inter,"voxUnion":union}

# ---------------------------
# IoU/Dice matrices
# ---------------------------
def build_mats(pred_vols, pred_meta, pred_names, gt_vols, gt_meta, gt_names):
    nP, nG = len(pred_vols), len(gt_vols)
    print(f"[MATRIX] {nP} x {nG}")
    bP = [world_bounds_um(v,m) for v,m in zip(pred_vols, pred_meta)]
    bG = [world_bounds_um(v,m) for v,m in zip(gt_vols,   gt_meta)]
    iou = np.zeros((nP,nG), dtype=np.float32)
    dice= np.zeros((nP,nG), dtype=np.float32)
    pairs = []  # (i,j,iou,dice)
    for i in range(nP):
        for j in range(nG):
            if not aabb_overlap_xy(bP[i], bG[j], margin_um=0.2):
                continue
            ij_iou, ij_dice, _ = iou_dice_pair(pred_vols[i], pred_meta[i], gt_vols[j], gt_meta[j])
            iou[i,j]  = ij_iou
            dice[i,j] = ij_dice
            pairs.append((pred_names[i], gt_names[j], ij_iou, ij_dice))
    return iou, dice, pairs

# ---------------------------
# Evaluate + save
# ---------------------------
def evaluate(pred_dir, gt_dir, out_dir, iou_thr=0.10):
    os.makedirs(out_dir, exist_ok=True)

    pred_vols, pred_meta, pred_names = load_vols_with_meta(pred_dir)
    gt_vols,   gt_meta,   gt_names   = load_vols_with_meta(gt_dir)
    if not pred_vols or not gt_vols:
        print("[ERR] missing data to compare.")
        return

    iou_mat, dice_mat, pairs = build_mats(pred_vols, pred_meta, pred_names, gt_vols, gt_meta, gt_names)

    # Hungarian matching on (1 - IoU)
    nP, nG = iou_mat.shape
    cost = 1.0 - iou_mat
    if nP > nG:
        pad = np.ones((nP, nP - nG), dtype=np.float32)
        cost = np.hstack([cost, pad])
    elif nG > nP:
        pad = np.ones((nG - nP, nG), dtype=np.float32)
        cost = np.vstack([cost, pad])
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < nP and c < nG:
            matches.append((r, c, float(iou_mat[r,c]), float(dice_mat[r,c])))

    TPs = [(r,c,iou,dice) for (r,c,iou,dice) in matches if iou >= iou_thr]
    matched_pred = {r for (r,c,_,_) in TPs}
    matched_gt   = {c for (r,c,_,_) in TPs}
    FPs = [i for i in range(nP) if i not in matched_pred]
    FNs = [j for j in range(nG) if j not in matched_gt]

    mean_iou  = float(np.mean([m[2] for m in TPs])) if TPs else 0.0
    mean_dice = float(np.mean([m[3] for m in TPs])) if TPs else 0.0
    prec = len(TPs) / max(1, (len(TPs)+len(FPs)))
    rec  = len(TPs) / max(1, (len(TPs)+len(FNs)))
    f1   = (2*prec*rec)/max(1e-8, (prec+rec)) if (prec+rec)>0 else 0.0

    # --- Save pair_stats.csv (all candidate pairs with overlap)
    pair_csv = os.path.join(out_dir, "pair_stats.csv")
    with open(pair_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pred","gt","iou","dice"])
        for pn, gn, i, d in pairs:
            w.writerow([pn, gn, f"{i:.6f}", f"{d:.6f}"])
    print(f"[OUT] {pair_csv}")

    # --- Save matches.csv (TPs plus FPs/FNs)
    match_csv = os.path.join(out_dir, "matches.csv")
    with open(match_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pred","gt","iou","dice","label"])
        for (r,c,i,d) in TPs:
            w.writerow([pred_names[r], gt_names[c], f"{i:.6f}", f"{d:.6f}", "TP"])
        for i in FPs:
            w.writerow([pred_names[i], "", "0.000000", "0.000000", "FP"])
        for j in FNs:
            w.writerow(["", gt_names[j], "0.000000", "0.000000", "FN"])
    print(f"[OUT] {match_csv}")

    # --- Save summary.json
    summary = {
        "pred_count": nP, "gt_count": nG,
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
    print(f"[OUT] {os.path.join(out_dir,'summary.json')}")

    # --- Console summary
    print("\n=== SUMMARY ===")
    print(f"Pred: {nP} | GT: {nG} | IoU_thr={iou_thr}")
    print(f"TP: {len(TPs)} | FP: {len(FPs)} | FN: {len(FNs)}")
    print(f"mean IoU:  {mean_iou:.4f}")
    print(f"mean Dice: {mean_dice:.4f}")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

# ---------------------------
# Main / CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute IoU/Dice between Pred and GT volumes")
    ap.add_argument("--pred_dir", type=str, default="", help="folder of predicted vol_*.npy/meta")
    ap.add_argument("--gt_dir",   type=str, default="", help="folder of GT vol_*.npy/meta")
    ap.add_argument("--out_dir",  type=str, default="", help="output folder for CSV/JSON")
    ap.add_argument("--iou_thr",  type=float, default=0.10, help="IoU threshold for TP")
    args = ap.parse_args()

    pred_dir = args.pred_dir or select_folder("Select PRED folder (vol_*.npy + _meta.json)")
    if not pred_dir: print("[ERR] no pred_dir"); sys.exit(1)
    gt_dir   = args.gt_dir   or select_folder("Select GT folder (vol_*.npy + _meta.json)")
    if not gt_dir: print("[ERR] no gt_dir"); sys.exit(1)

    # Ask where to save outputs (if not provided on CLI)
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = select_folder("Select OUTPUT folder for CSV/JSON (will be created if missing)")
        if not out_dir:
            out_dir = pred_dir
            print(f"[INFO] No output folder selected; using pred_dir: {out_dir}")

    print(f"[CFG] pred_dir={pred_dir}")
    print(f"[CFG] gt_dir  ={gt_dir}")
    print(f"[CFG] out_dir ={out_dir}")
    print(f"[CFG] IoU_thr ={args.iou_thr}")

    evaluate(pred_dir, gt_dir, out_dir, iou_thr=args.iou_thr)

if __name__ == "__main__":
    main()
