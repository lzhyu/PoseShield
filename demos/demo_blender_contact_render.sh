#!/bin/bash

set -euo pipefail

BLENDER_PATH="${BLENDER_PATH:-blender}"
FFMPEG_PATH="${FFMPEG_PATH:-ffmpeg}"
DEMO_DIR="demos/contact_render_demo"
OUTPUT_PATH="${DEMO_DIR}/render_contact.mp4"
CONTACT_MASK_PATH="${CONTACT_MASK_PATH:-${DEMO_DIR}/contact_masks.npz}"

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
    --original "${DEMO_DIR}/original_motion.npy" \
    --optimized "${DEMO_DIR}/optimized_motion.npy" \
    --output "$OUTPUT_PATH" \
    --blender-path "$BLENDER_BIN" \
    --ffmpeg-path "$FFMPEG_PATH" \
    --fps 15 \
    --samples 32 \
    --engine BLENDER_EEVEE \
    --frame-stride 2 \
    --res-x 1280 \
    --res-y 720 \
    --highlight-contact \
    --contact-mask-path "$CONTACT_MASK_PATH"

echo "Rendered contact-overlay demo to: $OUTPUT_PATH"
