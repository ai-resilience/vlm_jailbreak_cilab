#!/bin/bash
# bash ./scripts/inference/run_figstep_font.sh
# Run inference on FigStep dataset with different font sizes

gpu_id="1"
models=("phi" "deepseek2")
max_new_tokens=128

# Base directories
# Script is in vlm_kihyun_workspace/vlm_jailbreak_cilab/scripts/inference/
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SCRIPT_PATH="$PROJECT_ROOT/scripts/inference/run_inference.py"
OUTPUT_DIR="$SCRIPT_DIR/../result/inference"

echo "🚀 Starting FigStep inference with different font sizes..."
echo "📂 Output directory: $OUTPUT_DIR"
echo "📝 Script path: $SCRIPT_PATH"
echo "🎨 Font sizes: ${font_sizes[*]}"
echo "🤖 Models: ${models[*]}"
echo ""

# Main loop
for model in "${models[@]}"; do
    if [ "$model" == "phi" ]; then
        font_sizes=(20 40 60 100 110)  # Font sizes to test
    elif [ "$model" == "deepseek2" ]; then
        font_sizes=(10 20 30 40 50 60 70)  # Font sizes to test
    fi
    for font_size in "${font_sizes[@]}"; do
        echo "📁 Processing: $model / Figstep_font (font_size: $font_size)"
        
        # Run inference
        CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT_PATH" \
            --model_name "$model" \
            --dataset Figstep_font \
            --font_size "$font_size" \
            --output_dir "$OUTPUT_DIR" \
            --max_new_tokens $max_new_tokens
        
        if [ $? -eq 0 ]; then
            echo "    ✅ Inference completed for $model/Figstep_font/font$font_size"
        else
            echo "    ❌ Inference failed for $model/Figstep_font/font$font_size"
        fi
        echo ""
    done
done

echo "✅ All inferences complete!"

