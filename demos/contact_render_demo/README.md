# Blender Contact Rendering Demo

This folder contains one small before/after motion pair and a precomputed
contact mask for validating Blender contact-overlay rendering without running
the full motion optimizer.

- `original_motion.npy`: red input motion in public format, shaped `[T, 135]`
  with 22 joints in 6D rotation followed by XYZ global translation.
- `optimized_motion.npy`: green PoseShield output motion in the same public
  format.
- `contact_masks.npz`: TOPO=40 exact-FCL face masks for the original motion.
- `contact_summary.json`: mask metadata and collision-frame summary.
- `render_meshes.npz`: precomputed red/green SMPL-H meshes plus the yellow
  contact masks for quick rendering.
- `render_contact_preview.webm`: pre-rendered preview of the demo output.

The contact mask is a prerequisite for yellow overlays. It must match the same
original motion, SMPL-H mesh topology, and topology threshold used at render
time; Blender only visualizes this precomputed mask.

## Bundled Mesh Render

Preview:

<video src="render_contact_preview.webm" controls muted loop width="720"></video>

Run from the repository root:

```bash
bash demos/demo_blender_contact_render.sh
```

The script writes `render_contact.mp4` in this folder. This path uses the
bundled `render_meshes.npz`, so it does not require SMPL-H body-model files,
torch, or smplx. If Blender is not on `PATH`, set `BLENDER_PATH=/path/to/blender`.
By default it renders the full bundled 60-frame demo clip. FFmpeg is required
for MP4 encoding; if it is not on `PATH`, set `FFMPEG_PATH=/path/to/ffmpeg`.

For a faster smoke test, override the script defaults:

```bash
BLENDER_PATH=/path/to/blender \
FFMPEG_PATH=/path/to/ffmpeg \
FRAME_STRIDE=4 MAX_FRAMES=6 SAMPLES=4 \
bash demos/demo_blender_contact_render.sh
```

For a cleaner render, increase the sample count:

```bash
BLENDER_PATH=/path/to/blender \
FFMPEG_PATH=/path/to/ffmpeg \
SAMPLES=32 RES_X=1280 RES_Y=720 \
bash demos/demo_blender_contact_render.sh
```

For a custom motion pair, call `tools/render_motion_blender.py` directly with
`--original`, `--optimized`, and `--output`. Add `--highlight-contact` only when
you also provide a matching `--contact-mask-path`.

## Motion-To-Mesh Render

To verify the public-format motion-to-mesh step, render from the bundled motion
files instead of the precomputed mesh package:

```bash
BLENDER_PATH=/path/to/blender \
FFMPEG_PATH=/path/to/ffmpeg \
bash demos/demo_motion_to_mesh_contact_render.sh
```

This path runs SMPL-H forward kinematics on `original_motion.npy` and
`optimized_motion.npy`, then renders the generated meshes with the bundled
`contact_masks.npz`. It requires the full PoseShield Python environment and the
SMPL-H body-model assets. By default it renders a short 12-frame validation clip;
set `MAX_FRAMES=` to render the full sequence.

## Regenerate The Contact Mask

To verify the mask-export step as well, regenerate the exact-FCL masks first.
This path requires the full PoseShield Python environment, the SMPL-H body-model
assets, and the repository's compact topology cache at
`deps/topology_distances_30_60.npz`.

```bash
python tools/export_motion_contact_masks.py \
    --motions demos/contact_render_demo/original_motion.npy \
    --output-dir /tmp/poseshield_contact_demo_masks \
    --topology-threshold 40 \
    --rings 1 \
    --device cpu
```

Then render with the regenerated mask:

```bash
python tools/render_motion_blender.py \
    --original demos/contact_render_demo/original_motion.npy \
    --optimized demos/contact_render_demo/optimized_motion.npy \
    --output demos/contact_render_demo/render_contact_regenerated.mp4 \
    --blender-path /path/to/blender \
    --ffmpeg-path /path/to/ffmpeg \
    --highlight-contact \
    --contact-mask-path /tmp/poseshield_contact_demo_masks/original_motion_contact_masks.npz \
    --frame-stride 4 \
    --max-frames 6 \
    --fps 10 \
    --samples 4 \
    --res-x 960 \
    --res-y 540
```

The yellow overlay is only applied to the red original motion. The green
PoseShield output is rendered without contact patches so the before/after
comparison stays readable.
