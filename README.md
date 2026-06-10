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
pip install -e '.[landmarks,notebook]'
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
