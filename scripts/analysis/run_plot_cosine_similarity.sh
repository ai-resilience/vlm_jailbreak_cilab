#!/bin/bash
# bash ./scripts/analysis/run_plot_cosine_similarity.sh
# Plot cosine similarity between image-only and text-only hidden states

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

# Models to process
MODELS=("intern")

# Datasets to process
DATASETS=("Figstep")

# Pooling modes: token (extract specific tokens) or mean_pooling (average across sequence)
POOLING_MODES=("mean_pooling")

# Token indices to analyze (only used when pooling_mode='token')
TOKEN_INDICES=(-1 -2 -3 -4 -5)

echo "=========================================="
echo "Plotting Cosine Similarity"
echo "=========================================="

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for pooling_mode in "${POOLING_MODES[@]}"; do
            if [ "$pooling_mode" == "token" ]; then
                # Process each token index
                for token_idx in "${TOKEN_INDICES[@]}"; do
                    echo ""
                    echo "Processing: $model / $dataset / Token $token_idx (pooling: $pooling_mode)"
                    echo "----------------------------------------"
                    
                    python scripts/analysis/plot_cosine_similarity_image_text.py \
                        --model_name "$model" \
                        --dataset "$dataset" \
                        --token_idx "$token_idx" \
                        --pooling_mode "$pooling_mode"
                    
                    if [ $? -eq 0 ]; then
                        echo "    ✅ Completed: $model / $dataset / Token $token_idx"
                    else
                        echo "    ❌ Failed: $model / $dataset / Token $token_idx"
                    fi
                done
            else
                # mean_pooling mode (no token index needed)
                echo ""
                echo "Processing: $model / $dataset / Mean Pooling"
                echo "----------------------------------------"
                
                python scripts/analysis/plot_cosine_similarity_image_text.py \
                    --model_name "$model" \
                    --dataset "$dataset" \
                    --pooling_mode "$pooling_mode"
                
                if [ $? -eq 0 ]; then
                    echo "    ✅ Completed: $model / $dataset / Mean Pooling"
                else
                    echo "    ❌ Failed: $model / $dataset / Mean Pooling"
                fi
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "All plots completed!"
echo "=========================================="

