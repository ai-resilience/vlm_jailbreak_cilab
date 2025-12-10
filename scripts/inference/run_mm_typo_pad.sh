#!/bin/bash
# bash ./scripts/inference/run_mm_typo_pad.sh
# Run inference on MM-SafetyBench TYPO images with white padding (run_mm_typo_pad.py)

gpu_id="1"
models=("llava_next" "qwen" "intern")  # 필요에 따라 모델 추가/변경  "deepseek2" "kimi"
max_new_tokens=128
subsets=("HateSpeech" "Illegal_Activitiy")  # 처리할 TYPO 서브셋

# Base directories
# Script is in vlm_kihyun_workspace/vlm_jailbreak_cilab/scripts/inference/
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SCRIPT_PATH="$PROJECT_ROOT/scripts/inference/run_mm_typo_pad.py"
RESIZED_DIR="$PROJECT_ROOT/dataset/MM_SafetyBench/TYPO_pad1024_to336"
OUTPUT_DIR="$SCRIPT_DIR/../result/inference"

echo "🚀 Starting MM-SafetyBench TYPO inference with padding..."
echo "📂 Resized dir : $RESIZED_DIR"
echo "📂 Output dir  : $OUTPUT_DIR"
echo "📝 Script path : $SCRIPT_PATH"
echo "🤖 Models     : ${models[*]}"
echo "📑 Subsets    : ${subsets[*]}"
echo ""

# Main loop
for model in "${models[@]}"; do
    echo "📁 Processing: $model"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT_PATH" \
        --model_name "$model" \
        --resized_dir "$RESIZED_DIR" \
        --output_dir "$OUTPUT_DIR/$model/mm_typo_pad" \
        --max_new_tokens $max_new_tokens \
        --subsets "${subsets[@]}"
    
    if [ $? -eq 0 ]; then
        echo "    ✅ Inference completed for $model"
    else:
        echo "    ❌ Inference failed for $model"
    fi
done

echo "✅ All inferences complete!"


