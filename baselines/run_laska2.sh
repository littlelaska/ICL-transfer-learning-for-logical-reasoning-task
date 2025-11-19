MODE="RAG"    # CoT/Direct/RAG/Logical
DATASET_NAME="FOLIO"
MODEL_NAME="qwen7"
SPLIT="dev"
LANGCHAIN_DB="gsm8k"
RAG_TOPK=10
DEMONSTRATION_NUM=4
ZERO_SHOT=false

RUN_CMD="python llms_baseline.py --model_name $MODEL_NAME --dataset_name $DATASET_NAME --split $SPLIT --mode $MODE --max_new_tokens 8192 --db_name $LANGCHAIN_DB --icl_num $DEMONSTRATION_NUM --top_k $RAG_TOPK --batch_test --batch_size 16 --use_vllm --all_data_switch"


EVA_CMD="python evaluation.py --dataset_name $DATASET_NAME --model_name $MODEL_NAME --split $SPLIT --mode $MODE  --db_name $LANGCHAIN_DB --icl_num $DEMONSTRATION_NUM"

if [ "$ZERO_SHOT" = true ] && [ "$MODE" != "RAG" ]; then
    RUN_CMD="$RUN_CMD --zero_shot"
    EVA_CMD="$EVA_CMD --zero_shot"
    
fi
echo "Running: $RUN_CMD"
CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN_CMD
echo "Running: $EVA_CMD"
$EVA_CMD

