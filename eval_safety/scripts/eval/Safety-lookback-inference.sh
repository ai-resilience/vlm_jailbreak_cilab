export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false


MODEL="qwen2.5-vl"  # qwen2.5-vl, qwen3-vl, internvl3.5
MODEL_SIZE="7B"
EVAL_DIR="/absolute/path/to/eval_safety"
DATA_DIR="$EVAL_DIR/eval_data"
DATASET="wemath,mathvista,mathverse,mathvision,GeoMath,figstep,MM-SafetyBench-Typo,MM-SafetyBench-SD,MM-SafetyBench-SD-TYPO"

SYSTEM_PROMPT='Also, verify your response against the image during generation. The final answer MUST BE put in \boxed{}, i.e., \\boxed{final answer}'
RESULTS_DIR="/path/to/results/lookback/safety/${MODEL}/${MODEL_SIZE}/${DATASET}"
mkdir -p "$RESULTS_DIR"

if [ "$MODEL" == "qwen2.5-vl" ]; then
    HF_MODEL="Qwen/Qwen2.5-VL-${MODEL_SIZE}-Instruct"
    MODEL_FAMILY="qwen"
elif [ "$MODEL" == "qwen3-vl" ]; then
    HF_MODEL="Qwen/Qwen3-VL-${MODEL_SIZE}-Instruct"
    MODEL_FAMILY="qwen"
elif [ "$MODEL" == "internvl3.5" ]; then
    HF_MODEL="OpenGVLab/InternVL3_5-${MODEL_SIZE}"
    MODEL_FAMILY="intern"
fi

cd "$EVAL_DIR"

python main.py \
  --model "$HF_MODEL" \
  --model-family "$MODEL_FAMILY" \
  --output-dir "$RESULTS_DIR" \
  --data-path "$DATA_DIR" \
  --datasets "$DATASET" \
  --tensor-parallel-size 1 \
  --system-prompt="$SYSTEM_PROMPT" \
  --min-pixels 262144 \
  --max-pixels 1000000 \
  --max-model-len 8192 \
  --temperature 0.0 \
  --version="figstep_usr"