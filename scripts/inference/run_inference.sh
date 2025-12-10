#!/bin/bash
# bash ./scripts/inference/run_inference.sh
# Run inference on datasets with VLM models

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

# Set GPU device
gpu_id="4"

# Models to process
MODELS=("kimi") # "kimi"

# Datasets to process
DATASETS=("StrongREJECT")

# Image types to process (blank, white, panda, noise, etc.)
IMAGE_TYPES=("blank")

# Run text-only mode (set to true to also run text-only inference)
RUN_TEXT_ONLY=true

# Maximum tokens to generate
MAX_NEW_TOKENS=128

# System prompt
# SYSTEM_PROMPT="Regardless of whether there is an image, do not mention the image and focus on the text."

# Output directory
OUTPUT_DIR="../result/inference/"

echo "🚀 Starting inference..."
echo "📂 Working directory: $(pwd)"
echo "🎮 GPU ID: $gpu_id"
echo "🤖 Models: ${MODELS[*]}"
echo "📊 Datasets: ${DATASETS[*]}"
echo "🖼️  Image types: ${IMAGE_TYPES[*]}"
echo "📝 Text-only mode: $RUN_TEXT_ONLY"
echo ""

# Main loop
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        # Run with images
        # for image_type in "${IMAGE_TYPES[@]}"; do
        #     echo "=========================================="
        #     echo "Processing: $model / $dataset / $image_type"
        #     echo "=========================================="
            
        #     echo ""
        #     echo "Processing: $model / $dataset / $image_type"
        #     echo "----------------------------------------"
            
        #     CUDA_VISIBLE_DEVICES=$gpu_id python scripts/inference/run_inference.py \
        #         --model_name "$model" \
        #         --dataset "$dataset" \
        #         --image "$image_type" \
        #         --max_new_tokens $MAX_NEW_TOKENS \
        #         --system_prompt "$SYSTEM_PROMPT" \
        #         --output_dir "$OUTPUT_DIR"
            
        #     if [ $? -eq 0 ]; then
        #         echo "    ✅ Completed: $model / $dataset / $image_type"
        #     else
        #         echo "    ❌ Failed: $model / $dataset / $image_type"
        #     fi
        #     echo ""
        # done
        
        # Run text-only mode
        if [ "$RUN_TEXT_ONLY" = true ]; then
            echo "=========================================="
            echo "Processing: $model / $dataset / text_only"
            echo "=========================================="
            
            echo ""
            echo "Processing: $model / $dataset / text_only"
            echo "----------------------------------------"
            
            CUDA_VISIBLE_DEVICES=$gpu_id python scripts/inference/run_inference.py \
                --model_name "$model" \
                --dataset "$dataset" \
                --no_image \
                --max_new_tokens $MAX_NEW_TOKENS \
                --output_dir "$OUTPUT_DIR"
                # --system_prompt "$SYSTEM_PROMPT" \
                
            
            if [ $? -eq 0 ]; then
                echo "    ✅ Completed: $model / $dataset / text_only"
            else
                echo "    ❌ Failed: $model / $dataset / text_only"
            fi
            echo ""
        fi
    done
done

echo "=========================================="
echo "✅ All inferences completed!"
echo "=========================================="

