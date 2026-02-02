#!/bin/bash
# bash ./scripts/analysis/run_plot_pca.sh
# Run PCA analysis on multiple data classes with different models and class combinations

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

# Set GPU device
gpu_id="2"

# Models to process
MODELS=("intern" "qwen")

# Class combinations to process
# Format: "modality:type:safety" (comma-separated for multiple classes)
# Leave empty to use all classes
CLASS_COMBINATIONS=(
    # ""  # All classes
    "image:figstep:safe,image:figstep:unsafe","text:figstep:safe,text:figstep:unsafe"
    # "text:safeguard:safe,text:safeguard:unsafe"
    # "image:laion_coco:safe,text:safeguard:safe"
    # "image:laion_coco:unsafe,text:safeguard:unsafe,image:ocr:unsafe"
)

# Maximum samples per class (None for all)
MAX_SAMPLES_PER_CLASS=100

# Output directory
OUTPUT_DIR="../result/pca"

echo "🚀 Starting PCA analysis..."
echo "📂 Working directory: $(pwd)"
echo "🎮 GPU ID: $gpu_id"
echo "🤖 Models: ${MODELS[*]}"
echo "📊 Class combinations: ${#CLASS_COMBINATIONS[@]} combinations"
echo "🖼️  Max samples per class: $MAX_SAMPLES_PER_CLASS"
echo ""

# Main loop
for model in "${MODELS[@]}"; do
    for classes in "${CLASS_COMBINATIONS[@]}"; do
        echo "=========================================="
        if [ -z "$classes" ]; then
            echo "Processing: $model / ALL CLASSES"
            class_arg=""
        else
            echo "Processing: $model / $classes"
            class_arg="--classes \"$classes\""
        fi
        echo "=========================================="
        
        # Build and execute command
        if [ -z "$classes" ]; then
            # No classes specified - use all classes
            CUDA_VISIBLE_DEVICES=$gpu_id python scripts/analysis/plot_pca.py \
                --model_name "$model" \
                --max_samples_per_class $MAX_SAMPLES_PER_CLASS \
                --output_dir "$OUTPUT_DIR"
        else
            # Specific classes specified
            CUDA_VISIBLE_DEVICES=$gpu_id python scripts/analysis/plot_pca.py \
                --model_name "$model" \
                --classes "$classes" \
                --max_samples_per_class $MAX_SAMPLES_PER_CLASS \
                --output_dir "$OUTPUT_DIR"
        fi
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully completed: $model / ${classes:-ALL CLASSES}"
        else
            echo "❌ Failed: $model / ${classes:-ALL CLASSES}"
        fi
        echo ""
    done
done

echo "🎉 All PCA analysis completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
