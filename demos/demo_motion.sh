#!/bin/bash

set -euo pipefail

CKPT_PATH="ckpts/tencent/HY-Motion-1.0-Lite"

if [ ! -d "$CKPT_PATH" ]; then
    echo "Error: HY-Motion checkpoint not found at: $CKPT_PATH"
    echo "Please download HY-Motion-1.0-Lite and place it under ckpts/tencent/."
    exit 1
fi

echo "Available motion demo samples in demo_asset/:"
echo "1) motion_sample1.npy"
echo "2) motion_sample2.npy"
echo "3) motion_sample3.npy"

read -r -p "Select a sample (1-3) [default: 2]: " selection
selection=${selection:-2}

case "$selection" in
    1) SAMPLE="motion_sample1.npy";;
    2) SAMPLE="motion_sample2.npy";;
    3) SAMPLE="motion_sample3.npy";;
    *) echo "Invalid selection. Using sample 2."; SAMPLE="motion_sample2.npy";;
esac

MOTION_FILE="demo_asset/$SAMPLE"
STEM="${SAMPLE%.npy}"
OUTPUT_ROOT="demos/output_motion"
STAGE1_DIR="$OUTPUT_ROOT/$STEM"
STAGE2_DIR="$OUTPUT_ROOT/${STEM}_stage2"

mkdir -p "$STAGE1_DIR" "$STAGE2_DIR"

echo "------------------------------------------------------------"
echo "Running PoseShield motion collision resolution"
echo "  Model:  $CKPT_PATH"
echo "  Motion: $MOTION_FILE"
echo "  Stage 1 output: $STAGE1_DIR"
echo "  Stage 2 output: $STAGE2_DIR"
echo "------------------------------------------------------------"

python -m poseshield.hymotion.dno.run_dno_stage1 \
    --model_path "$CKPT_PATH" \
    --motion_file "$MOTION_FILE" \
    --output_dir "$STAGE1_DIR"

python -m poseshield.hymotion.dno.run_dno_stage2 \
    --model_path "$CKPT_PATH" \
    --motion_file "$MOTION_FILE" \
    --stage1_z "$STAGE1_DIR/stage1_z.pt" \
    --output_dir "$STAGE2_DIR"

python tools/generate_motion_html.py \
    --sequence "$STEM" \
    --original "$MOTION_FILE" \
    --optimized "$STAGE2_DIR/optimized_motion.npy" \
    --output-dir "$STAGE2_DIR/visualization"

echo "------------------------------------------------------------"
echo "Done. Key outputs:"
echo "  $STAGE1_DIR/stage1_z.pt"
echo "  $STAGE2_DIR/optimized_motion.npy"
echo "  $STAGE2_DIR/optimized_z.pt"
echo "  $STAGE2_DIR/summary.json"
echo "  $STAGE2_DIR/visualization/${STEM}_original_vs_optimized_vis.html"
echo "------------------------------------------------------------"
