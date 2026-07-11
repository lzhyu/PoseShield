# Blender Contact Rendering Demo

This folder contains one small before/after motion pair and a precomputed
contact mask for validating Blender contact-overlay rendering without running
the full motion optimizer.

- `original_motion.npy`: red input motion.
- `optimized_motion.npy`: green PoseShield output motion.
- `contact_masks.npz`: TOPO=40 exact-FCL face masks for the original motion.
- `contact_summary.json`: mask metadata and collision-frame summary.
- `render_meshes.npz`: precomputed red/green SMPL-H meshes plus the yellow
  contact masks for quick rendering.

The contact mask is a prerequisite for yellow overlays. It must match the same
original motion, SMPL-H mesh topology, and topology threshold used at render
time; Blender only visualizes this precomputed mask.

## Quick Render

Run from the repository root:

```bash
bash demos/demo_blender_contact_render.sh
```

The script writes `render_contact.mp4` in this folder. The quick path uses the
bundled `render_meshes.npz`, so it does not require SMPL-H body-model files,
torch, or smplx. If Blender is not on `PATH`, set `BLENDER_PATH=/path/to/blender`.

## Regenerate The Contact Mask

To verify the mask-export step as well, regenerate the exact-FCL masks first.
This uses the repository's compact topology cache at
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
    --highlight-contact \
    --contact-mask-path /tmp/poseshield_contact_demo_masks/original_motion_contact_masks.npz \
    --frame-stride 2 \
    --fps 15 \
    --res-x 1280 \
    --res-y 720
```

The yellow overlay is only applied to the red original motion. The green
PoseShield output is rendered without contact patches so the before/after
comparison stays readable.
