# BIMAP-P2---Bacteria-Shape-recognition

Bacteria 2D to 3D Pipeline (Prediction, Ground Truth, Visualization, Metrics)


# Overview of the Pipeline

* 2D segmentation & contours
Step1_bacteria_segmentation_and 2D contours creator.py
→ produces per-instance 2D masks/contours for predictions.

* 2D → solid 3D (predictions)
Step2_2D contours to 3D npy volumes transformer.py
→ PCA (major axis) + 360° rotation → solid 3D volume
→ outputs vol_*.npy + vol_*_meta.json (Pred).

* 3D visualization (overlay + background)
Step3_Step3_3D visualisation.py
→ Visualize the 3D volumes (Predicted or Groundtruth)
  
* Ground truth from ROIset + CZI
Step4_ Groundtruth ROIset to npy volumes.py
→ fills polygons per slice, closes 2D holes, reads CZI scaling
→ sets Z so the occupied GT mid aligns near Z ≈ 0
→ outputs vol_*.npy + vol_*_meta.json (GT).

* Step5_3D visualisation and overlay of predicted and groundtrouth volumes.py
Pred = white, GT = magenta
Background image (CZI/TIF/PNG/…) is mapped onto a plane at Z = 0 in microns.
Metrics (batch)

* Step6_ global 3D metrics calculation.py
Builds candidate pairs via XY AABB overlap
Resamples each pair on a shared local grid in world units
Computes IoU and Dice, then Hungarian matching (default IoU threshold = 0.10)
Outputs to chosen folder:
pair_stats.csv, matches.csv, summary.json
Metrics (interactive, per-pair)

* Step7_ one by one 3D metrics calculation.py
Runs batch metrics (saves CSV/JSON)
Visualizes overlay with labels (IDs = vol_XXXX)
Prompts for Pred ID + GT ID to compute IoU/Dice on demand
Also asks where to save CSV results.


# End-to-End tools to:

segment bacteria in 2D,
Transform 2D contours to solid 3D volumes (by rotating around PCA major axis),
convert ImageJ ROIset (ZIP) + .CZI z-stack to the same 3D volume format,
visualize predicted vs. ground-truth volumes over a background plane,
compute IoU and Dice (batch + interactive per-pair).
All per-instance outputs are standardized as:
vol_XXXX.npy — binary 3D array (Z, Y, X)
vol_XXXX_meta.json — metadata with physical units (microns) and global placement

# Requirements

Python 3.9–3.11
Recommended: GPU for Cellpose (optional)
Necessary packages:
pip install numpy scipy scikit-image scikit-learn pyvista aicsimageio czifile matplotlib
pip install cellpose torch

choose the right torch build for your system (GPU/CPU)

# Output Format (standardized)

Every instance (bacterium) is saved as:
vol_XXXX.npy: binary volume (Z, Y, X) with 0/1
vol_XXXX_meta.json with keys:
voxel_size_xy_um, voxel_size_z_um
offset_xy_global_px (global XY placement in pixels)
z_min_um, z_max_um, z0_index
volume_shape_zyx, instance_id, source_image, is_gt
This guarantees correct global placement and consistent microns-scale rendering.
