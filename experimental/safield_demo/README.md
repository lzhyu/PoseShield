# Shape-Aware Collision Field Experimental Demo

This directory contains a minimal experimental shape-aware collision-field demo.
Given a fixed colliding SMPL-H pose and two SMPL body-shape coefficient vectors,
the demo loads the released checkpoint, resolves the pose for each shape, and
optionally exports the input and resolved meshes as OBJ files.

## Setup

Use the PoseShield release environment from the repository root:

```bash
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
3115616ae89785ef3dd2343a56689672d0fd27810c796f7545f3c4f78bf3997b
```

For OBJ export, install the SMPL-H body model in the layout described by
`../../README.md`:

```text
deps/body_models/smplh/SMPLH_NEUTRAL.npz
```

## Run The Demo

Run the selected shape-aware optimization example:

```bash
python experimental/safield_demo/run_demo.py \
  --config_path experimental/safield_demo/config.yaml \
  --examples_path experimental/safield_demo/demo_examples.json \
  --example_idx 0
```

This prints the SAField values before and after optimization for both body
shapes.

To also export OBJ meshes:

```bash
python experimental/safield_demo/run_demo.py \
  --config_path experimental/safield_demo/config.yaml \
  --examples_path experimental/safield_demo/demo_examples.json \
  --example_idx 0 \
  --smpl_model_path deps/body_models \
  --output_dir experimental/safield_demo/output/example_0
```

The OBJ export writes:

```text
experimental/safield_demo/output/example_0/shape_a_input.obj
experimental/safield_demo/output/example_0/shape_a_resolved.obj
experimental/safield_demo/output/example_0/shape_b_input.obj
experimental/safield_demo/output/example_0/shape_b_resolved.obj
```

The output directory is ignored by Git.
