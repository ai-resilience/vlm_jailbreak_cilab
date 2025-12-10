#!/bin/bash
# bash ./scripts/eval/run_ensemble.sh
# Ensemble evaluation script for 3 metrics (beaverdam, llamaguard4, wildguard)
# Automatically finds evaluation result files and creates ensemble results

gpu_id="0"
models=("kimi") # "llava" "llava_next" "qwen" "deepseek2" "kimi"
datasets=("mm_sd") # "Figstep" "Figstep_font" "Figstep_res"

# Base directories
EVALUATION_DIR="../result/evaluation"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_PATH="$PROJECT_ROOT/scripts/eval/run_ensemble.py"

echo "🚀 Starting ensemble evaluation..."
echo "📂 Evaluation directory: $EVALUATION_DIR"
echo "📝 Script path: $SCRIPT_PATH"
echo ""

# Function to check if all 3 metric files exist
check_metric_files() {
    local base_name="$1"
    local eval_dir="$2"
    
    local beaverdam_file="${eval_dir}/${base_name}_beaverdam.jsonl"
    local llamaguard4_file="${eval_dir}/${base_name}_llamaguard4.jsonl"
    local wildguard_file="${eval_dir}/${base_name}_wildguard.jsonl"
    
    if [ -f "$beaverdam_file" ] && [ -f "$llamaguard4_file" ] && [ -f "$wildguard_file" ]; then
        echo "$beaverdam_file|$llamaguard4_file|$wildguard_file"
        return 0
    else
        return 1
    fi
}

# Main loop
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        eval_dir="$EVALUATION_DIR/$model/$dataset"
        
        # Skip if directory doesn't exist
        if [ ! -d "$eval_dir" ]; then
            echo "⚠️  Directory not found: $eval_dir"
            continue
        fi
        
        echo "📁 Processing: $model/$dataset"
        
        # Find all beaverdam evaluation files
        for beaverdam_file in "$eval_dir"/*_beaverdam.jsonl; do
            # Skip if no files found
            [ -f "$beaverdam_file" ] || continue
            
            filename=$(basename "$beaverdam_file")
            
            # Extract base name (remove _beaverdam.jsonl suffix)
            base_name="${filename%_beaverdam.jsonl}"
            
            # Skip if already has ensemble result
            ensemble_file="${eval_dir}/${base_name}_ensemble.jsonl"
            # if [ -f "$ensemble_file" ]; then
            #     echo "  ⏭️  Skipping $base_name (ensemble already exists)"
            #     continue
            # fi
            
            # Check if all 3 metric files exist
            if check_metric_files "$base_name" "$eval_dir" > /dev/null; then
                metric_files=$(check_metric_files "$base_name" "$eval_dir")
                IFS='|' read -r beaverdam_path llamaguard4_path wildguard_path <<< "$metric_files"
                
                echo "  📄 Processing: $base_name"
                
                # Run ensemble
                python "$SCRIPT_PATH" \
                    --beaverdam "$beaverdam_path" \
                    --llamaguard4 "$llamaguard4_path" \
                    --wildguard "$wildguard_path" \
                    --output "$ensemble_file"
                
                if [ $? -eq 0 ]; then
                    echo "  ✅ Ensemble completed for $base_name"
                else
                    echo "  ❌ Ensemble failed for $base_name"
                fi
            else
                echo "  ⚠️  Missing metric files for $base_name"
            fi
        done
        echo ""
    done
done

echo "✅ All ensemble evaluations complete!"

