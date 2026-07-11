#!/bin/bash

set -euo pipefail

BLENDER_PATH="${BLENDER_PATH:-blender}"
FFMPEG_PATH="${FFMPEG_PATH:-ffmpeg}"
DEMO_DIR="demos/contact_render_demo"
OUTPUT_PATH="${DEMO_DIR}/render_contact.mp4"
MESH_PATH="${MESH_PATH:-${DEMO_DIR}/render_meshes.npz}"

if command -v "$BLENDER_PATH" >/dev/null 2>&1; then
    BLENDER_BIN="$BLENDER_PATH"
elif [ -x "$BLENDER_PATH" ]; then
    BLENDER_BIN="$BLENDER_PATH"
else
    echo "Error: Blender executable not found at: $BLENDER_PATH"
    echo "Set BLENDER_PATH first, for example:"
    echo "  BLENDER_PATH=/path/to/blender bash demos/demo_blender_contact_render.sh"
    exit 1
fi

python tools/render_motion_blender.py \
    --mesh-path "$MESH_PATH" \
    --output "$OUTPUT_PATH" \
    --blender-path "$BLENDER_BIN" \
    --ffmpeg-path "$FFMPEG_PATH" \
    --fps 15 \
    --samples 32 \
    --engine BLENDER_EEVEE \
    --res-x 1280 \
    --res-y 720

echo "Rendered contact-overlay demo to: $OUTPUT_PATH"
