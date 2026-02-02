#!/bin/bash
# bash ./scripts/analysis/run_plot_pca_anchor.sh
# Run PCA analysis with anchor space projection
# Uses "text:safeguard:safe,text:safeguard:unsafe" as anchor to build PCA space,
# then projects other classes into this anchor space

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

# Set GPU device
gpu_id="2"

# Models to process
MODELS=("intern" "qwen")

# Maximum samples per class (None for all)
MAX_SAMPLES_PER_CLASS=100

# Output directory
OUTPUT_DIR="../result/pca"

echo "🚀 Starting PCA analysis with anchor space projection..."
echo "📂 Working directory: $(pwd)"
echo "🎮 GPU ID: $gpu_id"
echo "🤖 Models: ${MODELS[*]}"
echo "📊 Anchor classes: text:safeguard:safe, text:safeguard:unsafe"
echo "🖼️  Max samples per class: $MAX_SAMPLES_PER_CLASS"
echo ""

# Main loop
for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Processing: $model"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/analysis/plot_pca_anchor.py \
        --model_name "$model" \
        --max_samples_per_class $MAX_SAMPLES_PER_CLASS \
        --output_dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed: $model"
    else
        echo "❌ Failed: $model"
    fi
    echo ""
done

echo "🎉 All PCA anchor analysis completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
