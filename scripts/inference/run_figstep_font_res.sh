#!/bin/bash
# bash ./scripts/inference/run_figstep_font_res.sh
# Run inference on FigStep dataset with model-specific resolution images from SafeBench

gpu_id="0"
models=("phi")  # Add or remove models as needed
max_new_tokens=128

# Base directories
# Script is in vlm_kihyun_workspace/vlm_jailbreak_cilab/scripts/inference/
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SCRIPT_PATH="$PROJECT_ROOT/scripts/inference/run_figstep_font_res.py"
OUTPUT_DIR="$SCRIPT_DIR/../result/inference/fistep_font_res"

echo "🚀 Starting FigStep inference with model-specific resolutions..."
echo "📂 Output directory: $OUTPUT_DIR"
echo "📝 Script path: $SCRIPT_PATH"
echo "🤖 Models: ${models[*]}"
echo ""

# Main loop
for model in "${models[@]}"; do
    echo "📁 Processing: $model"
    
    # Run inference (resolutions are automatically handled by the script)
    CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT_PATH" \
        --model_name "$model" \
        --output_dir "$OUTPUT_DIR" \
        --max_new_tokens $max_new_tokens
    
    if [ $? -eq 0 ]; then
        echo "    ✅ Inference completed for $model"
    else
        echo "    ❌ Inference failed for $model"
    fi
    echo ""
done

echo "✅ All inferences complete!"

