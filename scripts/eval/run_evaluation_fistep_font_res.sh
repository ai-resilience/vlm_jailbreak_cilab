#!/bin/bash
# bash ./scripts/eval/run_evaluation_fistep_font_res.sh
# Evaluation script for fistep_font_res inference results
# Evaluates all resolution variations

gpu_id="2"
models=("phi") # "llava" "llava_next" "qwen" "deepseek2" "kimi" "intern"
metrics=("llamaguard4" "beaverdam" "wildguard") # "llamaguard4"

# Base directories
INFERENCE_DIR="../result/inference/fistep_font_res"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/eval/run_evaluation.py"
DATASET="Figstep_res"

echo "🚀 Starting evaluation of fistep_font_res inference results..."
echo "📂 Inference directory: $INFERENCE_DIR"
echo "📝 Script path: $SCRIPT_PATH"
echo "📊 Dataset: $DATASET"
echo ""

# Function to extract resolution from filename
# Pattern: {model}_Figstep_size{resolution}_response.jsonl
extract_resolution() {
    local filename="$1"
    # Extract resolution from pattern: size{resolution}
    echo "$filename" | sed -E 's/.*size([0-9]+)_.*/\1/'
}

# Function to check if file is already evaluated
is_evaluated() {
    local input_file="$1"
    local metric="$2"
    local model="$3"
    local base_name=$(basename "$input_file" .jsonl)
    local eval_file="../result/evaluation/${model}/${DATASET}/${base_name}_${metric}.jsonl"
    [ -f "$eval_file" ]
}

# Main loop
for model in "${models[@]}"; do
    model_dir="$INFERENCE_DIR/$model"
    
    # Skip if directory doesn't exist
    if [ ! -d "$model_dir" ]; then
        echo "⚠️  Directory not found: $model_dir"
        continue
    fi
    
    echo "📁 Processing: $model"
    
    # Create output directory
    output_dir="../result/evaluation/$model/${DATASET}"
    mkdir -p "$output_dir"
    
    # Find all response files for fistep_font_res
    for jsonl_file in "$model_dir"/*_Figstep_size*_response.jsonl; do
        # Skip if no files found
        [ -f "$jsonl_file" ] || continue
        
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

echo "✅ All fistep_font_res evaluations complete!"

