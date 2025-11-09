MODE="RAG"    # cot/direct/RAG/logical
DATASET_NAME="gsm8k"
MODEL_NAME="qwen7"
ZERO_SHOT=true
SYSTEM_PROMPT_PATH="./system_prompt"   # 目前除了logical模式，其他mode都不生效
PROMPT_FILE="logical_prompt_1.txt"     # 目前除了logical模式，其他mode都不生效
SPLIT="test"
LANGCHAIN_DB="logicaldeduction"
RAG_TOPK=10
DEMONSTRATION_NUM=1

RUN_CMD="python llms_baseline.py --model_name $MODEL_NAME --dataset_name $DATASET_NAME --split $SPLIT --mode $MODE --max_new_tokens 8192 --system_prompt_path  $SYSTEM_PROMPT_PATH --prompt_file $PROMPT_FILE --db_name $LANGCHAIN_DB --icl_num $DEMONSTRATION_NUM --top_k $RAG_TOPK --batch_test --batch_size 16 --use_vllm --all_data_switch"


EVA_CMD="python evaluation.py --dataset_name $DATASET_NAME --model_name $MODEL_NAME --split $SPLIT --mode $MODE  --db_name $LANGCHAIN_DB --icl_num $DEMONSTRATION_NUM"

if [ "$ZERO_SHOT" = true ]; then
    RUN_CMD="$RUN_CMD --zero_shot"
    EVA_CMD="$EVA_CMD --zero_shot"
    
fi
echo "Running: $RUN_CMD"
CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN_CMD
echo "Running: $EVA_CMD"
$EVA_CMD

