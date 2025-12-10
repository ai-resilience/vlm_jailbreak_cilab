#!/bin/bash
# bash ./scripts/eval/run_evaluation_figstep_font.sh
# Evaluation script for Figstep_font inference results
# Evaluates all font size variations

gpu_id="0"
models=("deepseek2") # "llava" "llava_next" "qwen" "deepseek2" "kimi"
images=("") # "panda" "noise"
modals=("image") # "text_only"
metrics=("llamaguard4" "beaverdam" "wildguard") # "llamaguard4"

if [ "$model" == "llava_next" ]; then
    font_sizes=(10 20 30 40 50 60 70) # Font sizes to evaluate
elif [ "$model" == "kimi" ]; then
    font_sizes=(20 40 60 100 110) # Font sizes to evaluate
elif [ "$model" == "intern" ]; then
    font_sizes=(20 40 60 100 110) # Font sizes to evaluate
elif [ "$model" == "phi" ]; then
    font_sizes=(20 40 60 100 110) # Font sizes to evaluate
elif [ "$model" == "qwen" ]; then
    font_sizes=(20 40 60 100 110) # Font sizes to evaluate
elif [ "$model" == "deepseek2" ]; then
    font_sizes=(50) # Font sizes to evaluate
fi

# Base directories
INFERENCE_DIR="../result/inference"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/eval/run_evaluation.py"
DATASET="Figstep_font"

echo "🚀 Starting evaluation of Figstep_font inference results..."
echo "📂 Inference directory: $INFERENCE_DIR"
echo "📝 Script path: $SCRIPT_PATH"
echo "📊 Dataset: $DATASET"
echo ""

# Function to extract font size from filename
# Pattern: {model}_{modality}_Figstep_font_font{size}_response.jsonl
extract_font_size() {
    local filename="$1"
    # Extract font size from pattern: font{size}
    echo "$filename" | sed -E 's/.*font([0-9]+)_.*/\1/'
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
    model_dir="$INFERENCE_DIR/$model/${DATASET}"
    
    # Skip if directory doesn't exist
    if [ ! -d "$model_dir" ]; then
        echo "⚠️  Directory not found: $model_dir"
        continue
    fi
    
    echo "📁 Processing: $model/$DATASET"
    
    # Create output directory
    output_dir="../result/evaluation/$model/${DATASET}"
    mkdir -p "$output_dir"
    
    # Find all response files for Figstep_font
    for jsonl_file in "$model_dir"/*_response.jsonl; do
        # Skip if no files found
        [ -f "$jsonl_file" ] || continue
        
        filename=$(basename "$jsonl_file")
        
        # Skip evaluation result files
        if [[ "$filename" == *"_llamaguard4_"* ]] || [[ "$filename" == *"_keyword_"* ]]; then
            continue
        fi
        
        # Extract font size from filename
        font_size=$(extract_font_size "$filename")
        
        # Skip if font size extraction failed
        if [ -z "$font_size" ] || [ "$font_size" == "$filename" ]; then
            echo "⚠️  Could not extract font size from: $filename"
            continue
        fi
        
        # Check if this font size is in our list (optional filter)
        # If font_sizes array is empty, process all files
        if [ ${#font_sizes[@]} -gt 0 ]; then
            font_found=false
            for fs in "${font_sizes[@]}"; do
                if [ "$font_size" == "$fs" ]; then
                    font_found=true
                    break
                fi
            done
            if [ "$font_found" == false ]; then
                echo "  ⏭️  Skipping font size $font_size (not in list)"
                continue
            fi
        fi
        
        echo "  📄 File: $filename (font size: $font_size)"
        
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

echo "✅ All Figstep_font evaluations complete!"

