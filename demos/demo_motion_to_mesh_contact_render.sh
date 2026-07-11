#!/bin/bash

set -euo pipefail

BLENDER_PATH="${BLENDER_PATH:-blender}"
FFMPEG_PATH="${FFMPEG_PATH:-ffmpeg}"
DEMO_DIR="demos/contact_render_demo"
ORIGINAL_MOTION="${ORIGINAL_MOTION:-${DEMO_DIR}/original_motion.npy}"
OPTIMIZED_MOTION="${OPTIMIZED_MOTION:-${DEMO_DIR}/optimized_motion.npy}"
CONTACT_MASK_PATH="${CONTACT_MASK_PATH:-${DEMO_DIR}/contact_masks.npz}"
OUTPUT_PATH="${OUTPUT_PATH:-${DEMO_DIR}/render_contact_motion_to_mesh.mp4}"
FPS="${FPS:-10}"
SAMPLES="${SAMPLES:-4}"
FRAME_STRIDE="${FRAME_STRIDE:-4}"
MAX_FRAMES="${MAX_FRAMES:-12}"
RES_X="${RES_X:-960}"
RES_Y="${RES_Y:-540}"
DEVICE="${DEVICE:-cpu}"

if command -v "$BLENDER_PATH" >/dev/null 2>&1; then
    BLENDER_BIN="$BLENDER_PATH"
elif [ -x "$BLENDER_PATH" ]; then
    BLENDER_BIN="$BLENDER_PATH"
else
    echo "Error: Blender executable not found at: $BLENDER_PATH"
    echo "Set BLENDER_PATH first, for example:"
    echo "  BLENDER_PATH=/path/to/blender bash demos/demo_motion_to_mesh_contact_render.sh"
    exit 1
fi

if command -v "$FFMPEG_PATH" >/dev/null 2>&1; then
    FFMPEG_BIN="$FFMPEG_PATH"
elif [ -x "$FFMPEG_PATH" ]; then
    FFMPEG_BIN="$FFMPEG_PATH"
else
    echo "Error: FFmpeg executable not found at: $FFMPEG_PATH"
    echo "Install FFmpeg or set FFMPEG_PATH first, for example:"
    echo "  FFMPEG_PATH=/path/to/ffmpeg bash demos/demo_motion_to_mesh_contact_render.sh"
    exit 1
fi

RENDER_ARGS=(
    tools/render_motion_blender.py
    --original "$ORIGINAL_MOTION"
    --optimized "$OPTIMIZED_MOTION"
    --output "$OUTPUT_PATH"
    --blender-path "$BLENDER_BIN"
    --ffmpeg-path "$FFMPEG_BIN"
    --device "$DEVICE"
    --highlight-contact
    --contact-mask-path "$CONTACT_MASK_PATH"
    --fps "$FPS"
    --samples "$SAMPLES"
    --engine BLENDER_EEVEE
    --frame-stride "$FRAME_STRIDE"
    --res-x "$RES_X"
    --res-y "$RES_Y"
)

if [ -n "$MAX_FRAMES" ]; then
    RENDER_ARGS+=(--max-frames "$MAX_FRAMES")
fi

python "${RENDER_ARGS[@]}"

echo "Rendered motion-to-mesh contact demo to: $OUTPUT_PATH"
