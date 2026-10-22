# 3DFaceReconstruction-With-Occlusion

Utilities and notebook experiments for reconstructing 3D faces when the input
image contains occlusions.  The original Colab export is preserved as
`new_(3) (8).py`; reusable code now lives under `src/face_recon_occlusion` so the
pipeline can be run, reviewed, and tested outside a notebook.

## What this project does

The workflow prepares face images for
[`Deep3DFaceRecon_pytorch`](https://github.com/sicxu/Deep3DFaceRecon_pytorch):

1. Generate Deep3DFaceRecon-compatible five-point landmarks.
2. Parse the face with BiSeNet.
3. Build an explicit non-face/occlusion mask.
4. Inpaint masked pixels with OpenCV Telea inpainting.
5. Run Deep3DFaceRecon on original and inpainted inputs.
6. Evaluate side-by-side render PNGs and reconstructed meshes.

> **Important:** The inpainting policy currently treats all non-face pixels as
> regions to inpaint.  This matches the notebook behavior, but it is a broad
> occlusion definition and should be interpreted as input normalization rather
> than precise object-level occlusion detection.

## Repository layout

```text
.
├── new_(3) (8).py                  # Original Colab-exported experiment log
├── pyproject.toml                  # Package metadata
├── requirements.txt                # Runtime dependencies
├── scripts/
│   ├── evaluate_2d.py              # CSV metrics for Deep3DFaceRecon PNG output
│   ├── generate_landmarks.py       # MTCNN landmark generation
│   └── inpaint_inputs.py           # BiSeNet + OpenCV inpainting CLI
└── src/face_recon_occlusion/
    ├── inpainting.py
    ├── landmarks.py
    ├── metrics_2d.py
    ├── metrics_3d.py
    └── segmentation.py
```

## Installation

Create an environment and install the local package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For the full landmark/notebook workflow, install optional dependencies:

```bash
pip install -e '.[landmarks,notebook,reconstruction]'
```

You still need external model assets that are not committed to this repository:

- A checkout of `Deep3DFaceRecon_pytorch`.
- The Basel Face Model files required by Deep3DFaceRecon.
- A Deep3DFaceRecon checkpoint such as `face_recon_feat0.2_augment`.
- A checkout of `face-parsing.PyTorch` that exposes `model.py`.
- BiSeNet face-parsing weights such as `79999_iter.pth`.

Large generated folders, checkpoints, meshes, and external repositories are
ignored by `.gitignore`.

## Typical workflow

### 1. Generate landmarks

Deep3DFaceRecon expects one text file per image under a `detections` folder.
Each file contains five points in this order: left eye, right eye, nose, left
mouth corner, right mouth corner.

```bash
python scripts/generate_landmarks.py \
  --image-dir /path/to/my_images
```

This writes landmark files to `/path/to/my_images/detections`.

### 2. Inpaint inputs

```bash
python scripts/inpaint_inputs.py \
  --input-dir /path/to/my_images \
  --output-dir /path/to/input_images_inpainted \
  --weights-path /path/to/face-parsing.PyTorch/res/cp/79999_iter.pth \
  --face-parsing-repo /path/to/face-parsing.PyTorch
```

The script loads BiSeNet, builds a binary face mask, inverts it into an OpenCV
inpainting mask, and saves inpainted copies using the original filenames.

### 3. Run Deep3DFaceRecon

Run Deep3DFaceRecon from its own checkout.  Example:

```bash
cd /path/to/Deep3DFaceRecon_pytorch
python test.py \
  --name face_recon_feat0.2_augment \
  --epoch 20 \
  --img_folder /path/to/input_images_inpainted \
  --phase test
```

### 4. Evaluate 2D render metrics

Deep3DFaceRecon writes combined PNG files where the first square is the input
and the second square is the rendered reconstruction.  Compute masked MSE, PSNR,
and SSIM with:

```bash
python scripts/evaluate_2d.py \
  --results-dir /path/to/results/epoch_20_000000 \
  --output-csv outputs/metrics_2d.csv
```

## Notes on metrics

- **MSE/PSNR/SSIM** compare the 2D input crop and rendered reconstruction. They
  are render-similarity metrics, not ground-truth 3D accuracy.
- **Mesh RMSE/NME/Chamfer** compare two reconstructed meshes. When comparing
  original vs. inpainted reconstructions, the values measure sensitivity to the
  preprocessing choice, not absolute reconstruction error.
- True 3D accuracy requires ground-truth scans or meshes.

## Development checks

The original Colab export contains notebook shell commands and is not expected
to pass `python -m py_compile`.  The reusable package and scripts should pass:

```bash
python -m compileall src scripts
```


## BTP Part 2 — geometry-based PAD pilot

The Phase 2 branch adds a reproducible feature-extraction and evaluation pipeline:
- Six mesh-level features: depth standard deviation/range, depth-gradient mean/std, and Gaussian-curvature mean/std.
- Subject-disjoint splits and five feature ablations using a StandardScaler + RBF-SVM baseline.
- PAD metrics including APCER, BPCER, ACER, ROC-AUC, F1 and confusion matrices.
- Failure analysis, nine numbered figures, strict JSON summary, final CSV and a Markdown report.
- A smoke test that executes the complete final-analysis command on explicitly labelled synthetic data.

The 100-sample pilot is **not an official OULU-NPU protocol evaluation**. Synthetic smoke-test metrics verify software execution only and must not be reported as research results.

### Corrected Colab run

Use the companion notebook `BTP_Part2_Geometry_PAD_OULU_corrected.ipynb` supplied with this change from a Colab runtime with Google Drive mounted. It checks out the `btp-phase2-corrected` branch and includes explicit checkpoints after each stage. It writes to a unique timestamped `phase2_corrected_runs/` folder and does not overwrite the original `oulu_geometry_features.csv`.

Expected experiment outputs:
- `oulu_geometry_features_corrected.csv`
- `ablation_results_corrected/results.csv` and `experiment_config.json`
- `final_analysis_corrected/final_results.csv`
- `final_analysis_corrected/final_experiment_summary.json`
- `final_analysis_corrected/FINAL_ANALYSIS.md`
- Nine figures in `final_analysis_corrected/final_figures/`

The notebook checks outputs after each stage and stops on a failed assertion or subprocess. The gradient-scale comparison should show that the corrected gradient no longer divides by `xy_scale`; other geometry features should stay unchanged within tolerance. Review extraction failures, subject-disjoint split checks, and the report before using results.

### Reconstruction-runtime preflight

The upstream Deep3DFaceRecon checkout imports both `kornia` and `dominate`.
Installing a package in one notebook/kernel interpreter does not prove that a
separately launched Python executable can import it. Check the exact interpreter
used for `test.py` before retrying an image:

```bash
python scripts/check_reconstruction_environment.py \
  --python /path/to/the/python-used-for-test.py \
  --reconstruction-repo /path/to/Deep3DFaceRecon_pytorch \
  --checkpoint /path/to/model-checkpoint.pth \
  --smoke-test
```

The helper checks `torch`, `torchvision`, `kornia`, and `dominate`. If
`kornia` or `dominate` is missing, it prints a package-install command using
the same interpreter; run that command and repeat the preflight. It does not
install packages automatically, and it never recommends blindly reinstalling
the core torch/torchvision stack. Add `--smoke-test` to run `test.py --help`,
which checks the import/CLI path without processing any images.

The geometry extractor now rejects invalid grid resolutions and malformed mesh
coordinates, verifies subject-disjoint splits before each ablation, writes cache
checkpoints through a temporary file replacement, and returns a non-zero exit
status if any mesh failed extraction. Partial progress is still saved for a safe
restart.

### Automated checks

GitHub Actions runs:

```bash
python -m compileall -q src scripts
python -m pytest -q
```

The end-to-end smoke test uses synthetic data and does not require Google Drive, OBJ pilot meshes, the Deep3DFaceRecon checkpoint, or the official OULU-NPU data. Passing CI therefore confirms code/output generation, not that the private Colab pilot has completed successfully.
