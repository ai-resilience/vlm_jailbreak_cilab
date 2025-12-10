#!/bin/bash
# bash ./scripts/eval/run_evaluation_padded.sh
# Evaluation script for padded inference results (mm_sd_pad, mm_typo_pad)
# Automatically finds and evaluates all padded inference result files

gpu_id="0"
models=("phi" "llava_next" "qwen" "deepseek2" "kimi" "intern") # "llava" "llava_next"   "qwen" "deepseek2" "kimi"
datasets=("mm_sd_pad") # Padded datasets
images=("") # "panda" "noise"
modals=("image") #  "text_only"
metrics=("wildguard" "llamaguard4" "beaverdam") # "llamaguard4"

# Base directories
INFERENCE_DIR="../result/inference"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/eval/run_evaluation.py"

echo "🚀 Starting evaluation of padded inference results..."
echo "📂 Inference directory: $INFERENCE_DIR"
echo "📝 Script path: $SCRIPT_PATH"
echo ""

# Function to extract resolution from filename
# Pattern: {model}_image_{dataset}_response_padded_{resolution}.jsonl
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
        # Padded files are in {model}/{dataset}/{model}/ directory
        model_dir="$INFERENCE_DIR/$model/$dataset/$model"

        # Skip if directory doesn't exist
        if [ ! -d "$model_dir" ]; then
            continue
        fi
        
        echo "📁 Processing: $model/$dataset"
        
        # Create output directory for this model/dataset
        output_dir="../result/evaluation/$model/$dataset"
        mkdir -p "$output_dir"
        
        # Find all padded response files
        # Pattern: {model}_image_{dataset}_response_padded_{resolution}.jsonl
        for jsonl_file in "$model_dir"/*_response_padded_*.jsonl; do
            # Skip if no files found
            [ -f "$jsonl_file" ] || continue
            
            # Skip if already evaluated
            filename=$(basename "$jsonl_file")
            
            # Skip evaluation result files
            if [[ "$filename" == *"_llamaguard4_"* ]] || [[ "$filename" == *"_keyword_"* ]] || [[ "$filename" == *"_ensemble_"* ]]; then
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

