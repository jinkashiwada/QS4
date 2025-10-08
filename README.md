# QS4: Quasi-Static Stabilization for Social Sensing
<p align="center">
  <video src="demo.mp4" autoplay loop muted playsinline width="720">
    Your browser does not support the video tag.
  </video>
</p>
This repository contains the source codes and example data used in the study submitted to the *International Journal of Disaster Risk Reduction (IJDRR)*.

---

## Overview
This study proposes the **Quasi-Static Stabilization for Social Sensing (QS4)** method, 
which enables the evaluation of water surface velocity distributions from unstable citizen video data.
The Python scripts demonstrate the processing pipeline, including:
- Feature detection and stabilization (Step 1)
- Homography-based orthorectification (Step 2)
- Optical flow analysis and visualization (Step 3)

---

## Current Status
**Status:** Under review  
**Submitted to:** *International Journal of Disaster Risk Reduction (IJDRR)*  
**Submission date:** October 2025  

The content of this repository may be updated after peer review or acceptance.  
Please note that the current version does not represent the final published work.

---

## How to Cite
If you refer to this work during the review period, please cite it as follows:

> Kashiwada, J. and Nihei, Y. (2025). Evaluation of Water Surface Velocity Distribution Using a Quasi-Static Stabilization Technique for Social Sensing. Manuscript submitted for publication.

A DOI and formal citation will be added after acceptance.

---

# Workflow

## Step 1 – Video Stabilisation

```bash
python 01_Step1_StabilizeVideo.py <video.mp4> --reference-frame <index> --output-scale <scale> [--roi X Y W H]
```
**Outputs (stored under `output/`):**
- `<video>_homography_preview.mp4` – stabilised preview video  
- `<video>_homography.csv` – per-frame projective transforms + match counts (`matches_total`, `matches_inliers`)  
- `01_match_viz/<video_stem>_matches.mp4` – reference vs current frame match visualisation  
- `01_QSIs/<video_stem>/frame_XXXX.png` – stabilised frames  
  (omit with `--skip-frame-export`, scale output via `--output-scale` if needed)

---

## Step 2 – Ortho-rectification Workflow

### 2-1 GCP Digitising
```bash
python 02-1_Step2_GCPCollector.py [--frames-dir output/01_QSIs/<video_stem>] [--frame-index N]
```
(Skip this step when a curated `GCPdata.csv` is supplied.)  
Select a representative stabilised frame, click GCPs, and populate `GCPdata.csv`.  
Annotated frames are written to `output/02-1_GCPs/<frame_name>/`.

---

### 2-2 Homography Calibration
```bash
python 02-2_Step2_CalcHomography.py --gcp GCPdata.csv [--enable-plot]
```
Produces:
- `output/02-2_transformed/homography_matrix.csv`  
- `output/02-2_transformed/homography_transformation_output.csv`  
- `output/02-2_transformed/overall_map.png`  
- Optional `residual_scatter.png` when `--enable-plot` is supplied

---

### 2-3 Apply Homography to Frames
```bash
python 02-3_Step2_ApplyHomography.py --gcp GCPdata.csv
```
Step 1 frames are discovered automatically under `output/01_QSIs/`  
(or specify `--frames-dir` to override).  
The first run creates `output/02_region_config.csv` (bounds + pix_per_meter); edit this file as required.  
Orthorectified PNGs are written to `output/02-3_Ortho-QSIs/<video_stem>/`.

---

### 2-4 Assemble Ortho Frames to Video
```bash
python 02-4_Step2_FramesToVideo.py
```
When multiple orthorectified sets exist, choose one from the prompt. Provide start/end indices and fps;  
the assembled MP4 is saved to `output/Ortho-QSI_<start>_<end>.mp4`.

---

## Step 3 – Optical Flow & Velocity Visualisation
```bash
python 03_Step3_OpticalFlow.py analyze --settings OpticalFlow_setting.csv [options]
python 03_Step3_OpticalFlow.py visualize --settings OpticalFlow_setting.csv [options]
```
- Reads `pix_per_meter` from `output/02_region_config.csv` (prompting if missing).  
- ROI selection dialog opens for confirmation; updates are saved back to `OpticalFlow_setting.csv`.  
- `analyze`: computes Farnebäck optical flow unless `--render-only` is supplied, then regenerates plots from existing CSVs.  
- `visualize`: generates plots from previously saved flow data.  

**Outputs per video (under `output/03_OpticalFlow/<stem>/`):**
- `optical_flow_overlay.mp4` – vectors on stabilised footage  
- `average_optical_flow_vectors.csv` – mean flow field in pixel units  
- `flow_heatmap.png` / `flow_heatmap_sketch.png` – heatmap (jet) + white vectors, scaled between 0 and ceil(95th percentile) unless `--heatmap-min/--heatmap-max` provided.

---

### Common Options
| Option | Description |
|--------|--------------|
| `--vector-step <int>` | spacing between drawn vectors (pixels) |
| `--line-thickness <int>` | vector line width (default 1) |
| `--heatmap-min/max <float>` | override automatic colour scaling |
| `--render-only` | skip re-computation and regenerate plots only |

---

## Data Preparation Notes
1. Populate `GCPdata.csv` with lat/lon (degrees) or planar x/y (meters) and mark `valid=1` for points to include in calibration.  
2. Ensure `pix_per_meter` in `output/02_region_config.csv` matches the intended scale.  
3. `OpticalFlow_setting.csv` should list `video_path` (e.g., `output/Ortho-QSI_0000_0158.mp4`). ROI columns will be created/updated by Step 3.

---

## Directory Layout
```
output/
  ├─ 01_QSIs/<video_stem>/                    – stabilised frames (Step 1)
  ├─ 01_match_viz/<video_stem>_matches.mp4    – match visualisation (Step 1)
  ├─ 02-1_GCPs/<frame_name>/                  – annotated GCP images (Step 2-1)
  ├─ 02_region_config.csv                     – ortho bounds & pix-per-meter (Step 2-3)
  ├─ 02-2_transformed/                        – homography outputs (Step 2-2)
  ├─ 02-3_Ortho-QSIs/<video_stem>/            – orthorectified frames (Step 2-3)
  ├─ Ortho-QSI_<start>_<end>.mp4              – ortho video(s) (Step 2-4)
  └─ 03_OpticalFlow/<video>/                  – optical flow results (Step 3)
```


---

## Beginner Demo – Guided Workflow
1. Stabilise your footage:
   ```bash
   python 01_Step1_StabilizeVideo.py Sample.mp4 --reference-frame 109 --output-scale 1.2
   ```
   Check `output/01_match_viz/` for the match animation and confirm frames appear under `output/01_QSIs/`.
2. (Optional) Review `GCPdata.csv` and adjust valid flags/coordinates if needed.  
3. Calibrate the homography:
   ```bash
   python 02-2_Step2_CalcHomography.py --gcp GCPdata.csv
   ```
   Inspect `output/02-2_transformed/` for diagnostics and the homography matrix file.
4. Apply the homography to the stabilised frames:
   ```bash
   python 02-3_Step2_ApplyHomography.py --gcp GCPdata.csv
   ```
   If prompted, supply `pix-per-meter` (e.g., 10); orthorectified frames appear under `output/02-3_Ortho-QSIs/Sample/`.
5. Convert the orthorectified frames to a video:
   ```bash
   python 02-4_Step2_FramesToVideo.py
   ```
   Select the desired frame set and provide start/end indices and frame rate. Output: `output/Ortho-QSI_<start>_<end>.mp4`.
6. (Optional) Run optical flow analysis and visualisation:
   ```bash
   python 03_Step3_OpticalFlow.py analyze --settings OpticalFlow_setting.csv
   python 03_Step3_OpticalFlow.py visualize --settings OpticalFlow_setting.csv
   ```

---

## License
This repository is distributed under the MIT License.  
Users are free to use, modify, and distribute the code with appropriate credit.

---

## Contact
**Author:** Jin Kashiwada  
**Affiliation:** Department of Civil Engineering, Tokyo University of Science  
**Email:** jin.kashiwada@rs.tus.ac.jp

---

*(This repository is made publicly available to promote reproducibility and transparency in hydrodynamic analysis research.)*
