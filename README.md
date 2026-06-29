# PoseShield: Neural Collision Fields for Human Self-Collision Resolution

<p align="center">
  Zhengyuan Li<sup>1</sup>&emsp;
  Zeyun Deng<sup>1</sup>&emsp;
  Yifan Shen<sup>3</sup>&emsp;
  <a href='https://cs.illinois.edu/about/people/faculty/lgui' target='_blank'>Liangyan Gui</a><sup>3</sup>&emsp;
  Miaolan Xie<sup>1</sup>&emsp;
  <br>
  Joseph Campbell<sup>1</sup>&emsp;
  Xifeng Gao<sup>2</sup>&emsp;
  Kui Wu<sup>2</sup>&emsp;
  Zherong Pan<sup>2</sup>&emsp;
  Aniket Bera<sup>1</sup>&emsp;
  <br><br>
  <sup>1</sup>Purdue University&emsp;
  <sup>2</sup>LightSpeed Studios&emsp;
  <sup>3</sup>University of Illinois Urbana-Champaign
  <br>
  <strong>ECCV 2026</strong>
</p>

<p align="center">
  <img src="assets/pipeline.png" alt="PoseShield Pipeline" width="90%">
  <br>
  <a href="assets/img_pipeline.pdf">Download the pipeline figure as PDF</a>
</p>

<p align="center">
  <video src="assets/PoseShield_demo.mp4" controls width="90%"></video>
</p>

<p align="center">
  <a href="assets/PoseShield_demo.mp4">Watch the PoseShield demo video</a>
</p>

**PoseShield** is a post-hoc self-collision resolver for SMPL-H poses and human motion sequences. It learns a neural collision field and uses it as a differentiable constraint for collision correction without retraining the upstream pose or motion generator.

## News

- **Jun 2026** — Initial code release with pre-trained models, pose-level optimization, and motion-level Stage 1/Stage 2 inference.

## Getting Started

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate poseshield
pip install -e .
```

We test the code on Python 3.10 and PyTorch with CUDA.

### 2. Download SMPL+H Body Models

Register and download the Extended SMPL+H model from the [MANO website](https://mano.is.tue.mpg.de/).

```bash
mkdir -p deps/body_models/smplh
cp smplh/neutral/model.npz deps/body_models/smplh/SMPLH_NEUTRAL.npz
```

PoseShield currently uses the neutral SMPL-H model.

### 3. Download Release Assets

Download and extract the PoseShield external assets at the repository root:

```bash
unzip PoseShield_release_dependencies_20260628.zip -d .
unzip PoseShield_release_pose_data_20260628.zip -d .
unzip PoseShield_release_motion_data_20260628.zip -d .
```

The three release asset packages are available from the
[PoseShield Google Drive folder](https://drive.google.com/drive/folders/1gLdFy4OTfYaKeaZ3olqShyh3kF2m5ogf?usp=sharing).

The dependency package provides the PoseShield checkpoints and exact-FCL mesh
distance table:

| File | Destination | Description |
|------|-------------|-------------|
| `model.pth` | `ckpts/poseshield/` | Collision field checkpoint |
| `config.yaml` | `ckpts/poseshield/` | Collision field config |
| `model_elu.pth` | `ckpts/poseshield/` | ELU collision field for motion resolution |
| `config_elu.yaml` | `ckpts/poseshield/` | ELU collision field config |
| `distances.pkl` | `deps/` | Mesh topology distances for exact-FCL checks |

The pose data package provides `data/dataset/` for pose model training and
evaluation, plus `data/dataset_test/` for the pose-level collision-resolution
benchmark. The motion data package provides the full 100-sample canonical
motion subset under `data/motion_canonical/`.

For motion-level resolution, also download [HY-Motion-1.0-Lite](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) and place it under:

```text
ckpts/tencent/HY-Motion-1.0-Lite/
```

This directory should contain the HY-Motion checkpoint, its config file, and normalization statistics:

```text
ckpts/tencent/HY-Motion-1.0-Lite/latest.ckpt
ckpts/tencent/HY-Motion-1.0-Lite/config.yaml  # or config.yml
ckpts/tencent/HY-Motion-1.0-Lite/stats/Mean.npy
ckpts/tencent/HY-Motion-1.0-Lite/stats/Std.npy
```

The repository includes small ready-to-run motion demos in `demo_asset/`. The full released canonical motion subset is distributed separately through the project release assets.

## Public Motion Format

All public motion inference, evaluation, and visualization code consumes the same canonical motion format:

```text
shape: [frames, 135]

[0:132]   22 joints × 6D rotations, HY-Motion column-interleaved layout
[132:135] absolute global translation [abs_x, abs_y, abs_z]
```

Coordinate convention:

```text
Y-up
X = right
Y = height/up
Z = forward
frame 0 human facing +Z
```

The same format is used for both input motions and `optimized_motion.npy`.

## Demo

### Pose Collision Resolution

```bash
python demos/demo_pose.py
```

Outputs are written to `demos/output/`.

### Motion Collision Resolution

Run the interactive demo:

```bash
bash demos/demo_motion.sh
```

Or run the two stages explicitly:

```bash
SAMPLE=motionfix_005334_135.npy
STEM=${SAMPLE%.npy}

python -m poseshield.hymotion.dno.run_dno_stage1 \
    --model_path ckpts/tencent/HY-Motion-1.0-Lite \
    --motion_file demo_asset/$SAMPLE \
    --output_dir demos/output_motion/$STEM

python -m poseshield.hymotion.dno.run_dno_stage2 \
    --model_path ckpts/tencent/HY-Motion-1.0-Lite \
    --motion_file demo_asset/$SAMPLE \
    --stage1_z demos/output_motion/$STEM/stage1_z.pt \
    --output_dir demos/output_motion/${STEM}_stage2
```

Stage 2 writes:

```text
optimized_motion.npy
optimized_z.pt
summary.json
args.json
```

The optimized motion copies the original absolute translation trajectory and updates only the pose rotations.

### Exact Mesh/FCL Collision Check

```bash
python tools/evaluate_exact_fcl.py \
    --motion demos/output_motion/${STEM}_stage2/optimized_motion.npy \
    --output-dir demos/output_motion/${STEM}_stage2/exact_fcl
```

If exact mesh collisions remain, the tool exits with a non-zero status after writing `exact_fcl_results.json`.

### HTML Visualization

```bash
python tools/generate_motion_html.py \
    --sequence $STEM \
    --original demo_asset/$SAMPLE \
    --optimized demos/output_motion/${STEM}_stage2/optimized_motion.npy \
    --output-dir demos/output_motion/${STEM}_stage2/visualization
```

Open the generated `*_vis.html` file in a browser.

### Optional Blender MP4 Rendering

For a higher-quality MP4 render, install Blender and FFmpeg, then run:

```bash
python tools/render_motion_blender.py \
    --original demo_asset/$SAMPLE \
    --optimized demos/output_motion/${STEM}_stage2/optimized_motion.npy \
    --output demos/output_motion/${STEM}_stage2/render.mp4 \
    --blender-path /path/to/blender
```

## Evaluation

The following pose-level evaluation commands require the released pose benchmark
split and the dependency package described above. The small files under
`demo_asset/` are intended for quick demos, not for reproducing the full
benchmark tables.

### Collision Detection Accuracy

```bash
python poseshield/pose/evaluate.py \
    --config-path ckpts/poseshield/config.yaml \
    --model-path ckpts/poseshield/model.pth
```

### Pose-Level Collision Resolution Benchmark

```bash
python poseshield/pose/resolve_dataset_test_slsqp.py \
    --config-path ckpts/poseshield/config.yaml \
    --model-path ckpts/poseshield/model.pth \
    --n-samples 500 \
    --cost-type weighted \
    --threshold 0.1 \
    --max-itr 200 \
    --save
```

## Training

Train the collision field from scratch:

```bash
python -m poseshield.pose.train --config-path config_files/basic_config.yaml
```

Checkpoints and logs are saved to `experiments/<EXP_NAME>/`.

## Acknowledgements

This project builds upon [SMPL-X](https://smpl-x.is.tue.mpg.de/), [HY-Motion-1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0), [python-fcl](https://github.com/BerkeleyAutomation/python-fcl), [Diffusion-Noise-Optimization](https://github.com/korrawe/Diffusion-Noise-Optimization), and the [MotionFix](https://motionfix.is.tue.mpg.de/) dataset.

## License

This project is licensed under the MIT License. External body models, datasets,
and upstream model checkpoints may be subject to their own licenses.
