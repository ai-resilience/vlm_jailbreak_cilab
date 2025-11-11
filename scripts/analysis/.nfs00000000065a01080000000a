#!/bin/bash
# bash ./scripts/analysis/run_hidden_states_with_res.sh
# Extract hidden states with resolution adjustment for multiple models/datasets

gpu_id="0"
models=("qwen") # "qwen" 
datasets=("mm_sd_typo") # "Figstep" "mm_sd_typo" "XSTest"

# Base directories
# Script is in vlm_kihyun_workspace/, project is in vlm_kihyun_workspace/vlm_jailbreak_cilab/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
SCRIPT_PATH="$PROJECT_ROOT/scripts/analysis/run_hidden_states_with_res.py"
OUTPUT_DIR="../result/hidden_states"

echo "🚀 Starting hidden state extraction with resolution adjustment..."
echo "📂 Output directory: $OUTPUT_DIR"
echo "📝 Script path: $SCRIPT_PATH"
echo ""

# Main loop
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "📁 Processing: $model/$dataset"
        
        # Run hidden state extraction
        CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT_PATH" \
            --model_name "$model" \
            --dataset "$dataset" \
            --output_dir "$OUTPUT_DIR"
        
        if [ $? -eq 0 ]; then
            echo "    ✅ Hidden state extraction completed for $model/$dataset"
        else
            echo "    ❌ Hidden state extraction failed for $model/$dataset"
        fi
        echo ""
    done
done

echo "✅ All hidden state extractions complete!"

