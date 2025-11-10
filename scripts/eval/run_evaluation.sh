#!/bin/bash
# bash ./scripts/eval/run_evaluation.sh
# Evaluation script for inference results
# Automatically finds and evaluates all inference result files

gpu_id="0"
models=("llava" "llava_next" "intern" "qwen" "deepseek2" "kimi")
datasets=("Figstep" "mm_sd_typo")
images=("blank") # "white" "panda" "noise"
modals=("image") #  "text_only"
metrics=("keyword") # "llamaguard4"

# Base directories
INFERENCE_DIR="../result/inference"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/eval/run_evaluation.py"

echo "🚀 Starting evaluation of inference results..."
echo "📂 Inference directory: $INFERENCE_DIR"
echo "📝 Script path: $SCRIPT_PATH"
echo ""

# Function to extract resolution from filename
# Pattern: {model}_{modality}_{dataset}_response_{image_type}_{resolution}.jsonl
extract_resolution() {
    local filename="$1"
    # Extract the last number before .jsonl
    echo "$filename" | sed -E 's/.*_([0-9]+)\.jsonl$/\1/'
}

# Function to check if file is already evaluated
is_evaluated() {
    local input_file="$1"
    local metric="$2"
    local model="$3"
    local dataset="$4"
    local base_name=$(basename "$input_file" .jsonl)
    local eval_file="../result/evaluation/${model}/${dataset}/${base_name}_${metric}.jsonl"
    [ -f "$eval_file" ]
}

# Main loop
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        model_dir="$INFERENCE_DIR/$model/$dataset"
        
        # Skip if directory doesn't exist
        if [ ! -d "$model_dir" ]; then
            continue
        fi
        
        echo "📁 Processing: $model/$dataset"
        
        # Create output directory for this model/dataset
        output_dir="../result/evaluation/$model/$dataset"
        mkdir -p "$output_dir"
        
        # Find all response files (excluding already evaluated ones)
        for jsonl_file in "$model_dir"/*_response_*.jsonl; do
            # Skip if no files found
            [ -f "$jsonl_file" ] || continue
            
            # Skip if already evaluated
            filename=$(basename "$jsonl_file")
            
            # Skip evaluation result files
            if [[ "$filename" == *"_llamaguard4_"* ]] || [[ "$filename" == *"_keyword_"* ]]; then
                continue
            fi
            
            # Extract resolution from filename
            resolution=$(extract_resolution "$filename")
            
            # Skip if resolution extraction failed
            if [ -z "$resolution" ] || [ "$resolution" == "$filename" ]; then
                echo "⚠️  Could not extract resolution from: $filename"
                continue
            fi
            
            echo "  📄 File: $filename (resolution: $resolution)"
            
            # Run evaluation for each metric
            for metric in "${metrics[@]}"; do
                # Check if already evaluated
                if is_evaluated "$jsonl_file" "$metric" "$model" "$dataset"; then
                    echo "    ⏭️  Skipping $metric (already evaluated)"
                    continue
                fi
                
                echo "    🔍 Evaluating with metric: $metric"
                
                # Run evaluation
                CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT_PATH" \
                    --input "$jsonl_file" \
                    --metric "$metric" \
                    --output_dir "$output_dir"
                
                if [ $? -eq 0 ]; then
                    echo "    ✅ $metric evaluation completed"
                else
                    echo "    ❌ $metric evaluation failed"
                fi
            done
        done
        echo ""
    done
done

echo "✅ All evaluations complete!"

