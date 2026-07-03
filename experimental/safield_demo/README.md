# Shape-Aware Collision Field (SAField) Experimental Demo

This directory contains the standalone experimental shape-aware collision demo
used by the PoseShield release README. It shows the same initial colliding
SMPL-H pose resolved under two different SMPL body-shape coefficient vectors.

The selected example is fixed in `demo_examples.json` and has exact-FCL
validation evidence:

- Shape A initial mesh: colliding
- Shape A resolved mesh: collision-free
- Shape B initial mesh: colliding
- Shape B resolved mesh: collision-free
- Shape coefficient range: each beta value is within `[-2, 2]`

## Release Assets

The release teaser image is:

```text
../../assets/safield_experimental_shape_demo_blender.png
```

It is generated from the four OBJ assets in:

```text
artifacts/release_example/release_objs/
```

Those OBJ files correspond to:

- `shape_a_input.obj`
- `shape_b_input.obj`
- `shape_a_resolved.obj`
- `shape_b_resolved.obj`

The full uncropped Blender render and `.blend` file are written under
`assets/` when the figure is regenerated from the repository root.

## Requirements

Use the PoseShield release environment:

```bash
cd /path/to/PoseShield
conda env create -f environment.yml
conda activate poseshield
pip install -e .
```

Download and extract the optional SAField demo asset package from the same
PoseShield Google Drive folder linked in `../../README.md`:

```bash
unzip PoseShield_release_safield_demo_20260703.zip -d .
```

This package installs:

```text
experimental/safield_demo/best_scc_model.pth
experimental/safield_demo/config.yaml
```

Expected SHA256 for `PoseShield_release_safield_demo_20260703.zip`:

```text
e8d3463b3ff9f9cac6473e10b30030c68fd611840566f97b636fe3918d532002
```

For mesh export and exact-FCL verification, install the SMPL-H body model in the
same layout described by `../../README.md`, for example:

```text
deps/body_models/smplh/SMPLH_NEUTRAL.npz
```

For Blender rendering, either put `blender` on `PATH` or pass its path through
the `BLENDER` environment variable.

## Quick Demo

Run the selected shape-aware optimization example:

```bash
cd /path/to/PoseShield
python experimental/safield_demo/run_demo.py \
  --config_path experimental/safield_demo/config.yaml \
  --examples_path experimental/safield_demo/demo_examples.json \
  --example_idx 0
```

This prints the SAField values before and after optimization for both shapes.
The `solver_success` flag is reported for diagnostics; the release example is
accepted by the field threshold and exact-FCL validation evidence below.

## Reproduce The Release Figure

If the OBJ assets are already present, render the README figure directly:

```bash
blender --background --python experimental/safield_demo/render_release_figure_blender.py -- \
  --obj-dir experimental/safield_demo/artifacts/release_example/release_objs \
  --output assets/safield_experimental_shape_demo_blender_full.png

python experimental/safield_demo/crop_release_figure.py \
  --input assets/safield_experimental_shape_demo_blender_full.png \
  --output assets/safield_experimental_shape_demo_blender.png \
  --top 0.03 \
  --bottom 1.0
```

The render command writes:

```text
assets/safield_experimental_shape_demo_blender_full.png
assets/safield_experimental_shape_demo_blender_full.blend
assets/safield_experimental_shape_demo_blender.png
```

The final README image is the cropped `safield_experimental_shape_demo_blender.png`.

## Regenerate The OBJ Assets

To regenerate the four Blender-compatible OBJ assets from `demo_examples.json`:

```bash
SMPL_MODEL_PATH=deps/body_models \
python experimental/safield_demo/export_release_objs.py \
  --examples-path experimental/safield_demo/demo_examples.json \
  --example-idx 0 \
  --output-dir experimental/safield_demo/artifacts/release_example/release_objs
```

`SMPL_MODEL_PATH` must be the parent directory that contains the `smplh/`
subdirectory.

## Exact-FCL Verification

The selected release example has already been verified with exact FCL. To rerun
the check:

```bash
SMPL_MODEL_PATH=deps/body_models \
DISTANCES_PATH=deps/distances.pkl \
python experimental/safield_demo/verify_selected_demo.py \
  --examples_path experimental/safield_demo/demo_examples.json \
  --example_idx 0 \
  --smpl_model_path "${SMPL_MODEL_PATH}" \
  --distances_path "${DISTANCES_PATH}" \
  --output_dir experimental/safield_demo/artifacts/release_example
```

The verification report is written to:

```text
experimental/safield_demo/artifacts/release_example/selected_fcl_report.json
```

Exact-FCL verification loads a large topology-distance table and should run in a
machine or allocation with sufficient RAM.

## Optional Candidate Search

The selected sample was found with a numeric search plus Qwen3-VL visual review.
The search script is included for reproducibility and further exploration:

```bash
SMPL_MODEL_PATH=deps/body_models \
python experimental/safield_demo/search_shape_conditioned_examples.py \
  --dataset /path/to/test_colliding_500.npz \
  --model_path experimental/safield_demo/best_scc_model.pth \
  --smpl_model_path "${SMPL_MODEL_PATH}" \
  --examples_path experimental/safield_demo/demo_examples.json \
  --output_dir experimental/safield_demo/artifacts \
  --max_poses 160 \
  --shape_pairs 10 \
  --top_k_render 24 \
  --threshold 0.05 \
  --max_itr 100 \
  --device cuda \
  --use_vlm
```

Qwen3-VL scores are used only as an auxiliary visual signal. The final selected
example should still be checked by exact FCL and by human visual review before
updating release assets.
