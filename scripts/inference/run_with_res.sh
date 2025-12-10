#!/bin/bash
# bash ./scripts/inference/run_with_res.sh
# Run inference with resolution adjustment for multiple models/datasets

gpu_id="0"
models=("phi" "llava_next" "qwen" "intern") 
datasets=("mm_sd") # "Figstep" "mm_sd_typo" "XSTest"
images=("") # "panda" "noise"
max_new_tokens=128

# Base directories
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/inference/run_with_res.py"
OUTPUT_DIR="../result/inference"

echo "🚀 Starting inference with resolution adjustment..."
echo "📂 Output directory: $OUTPUT_DIR"
echo "📝 Script path: $SCRIPT_PATH"
echo ""

# Main loop
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for image in "${images[@]}"; do
            echo "📁 Processing: $model/$dataset (image: $image)"
            
            # Run inference
            CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT_PATH" \
                --model_name "$model" \
                --dataset "$dataset" \
                --image "$image" \
                --output_dir "$OUTPUT_DIR" \
                --max_new_tokens $max_new_tokens
            
            if [ $? -eq 0 ]; then
                echo "    ✅ Inference completed for $model/$dataset/$image"
            else
                echo "    ❌ Inference failed for $model/$dataset/$image"
            fi
            echo ""
        done
    done
done

echo "✅ All inferences complete!"

