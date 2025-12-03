MODE="RAG"    # CoT/Direct/RAG/Logical
DATASET_NAME="AR-LSAT"    # gsm8k/ProntoQA/AR-LSAT/FOLIO/ProofWriter/LogicalDeduction
MODEL_NAME="qwen7"   # qwen14/qwen7/qwen3-8
SPLIT="test"
LANGCHAIN_DB="FOLIO"    # gsm8k//ProntoQA/FOLIO/ProofWriter/LogicalDeduction
DB_TYPE="embedding"   # bm25/embedding
RAG_TOPK=10
DEMONSTRATION_NUM=4
ZERO_SHOT=true
DTYPE="float16"
REVERSE_FLAG=false

LANGCHAIN_CMD="python dataset_cons.py --dataset_name $LANGCHAIN_DB --db_name $LANGCHAIN_DB --db_type $DB_TYPE --top_k $RAG_TOPK  --ds_cot"

RUN_CMD="python llms_baseline.py --model_name $MODEL_NAME --dataset_name $DATASET_NAME --split $SPLIT --mode $MODE --max_new_tokens 8192 --db_name $LANGCHAIN_DB --icl_num $DEMONSTRATION_NUM --top_k $RAG_TOPK --db_type $DB_TYPE --dtype $DTYPE --batch_test --batch_size 8 --use_vllm --all_data_switch"

EVA_CMD="python evaluation.py --dataset_name $DATASET_NAME --model_name $MODEL_NAME --split $SPLIT --mode $MODE  --db_name $LANGCHAIN_DB --icl_num $DEMONSTRATION_NUM"

if [ "$ZERO_SHOT" = true ] && [ "$MODE" != "RAG" ]; then
    RUN_CMD="$RUN_CMD --zero_shot"
    EVA_CMD="$EVA_CMD --zero_shot"
    
fi

if [ "$REVERSE_FLAG" = true ] && [ "$MODE" = "RAG" ] && [ "$DEMONSTRATION_NUM" > 1 ]; then
    RUN_CMD="$RUN_CMD --reverse_rag_order" 
fi

# echo "Building the langchain_dataset, Running: $LANGCHAIN_CMD"
# $LANGCHAIN_CMD

echo "Running: $RUN_CMD"
CUDA_VISIBLE_DEVICES=0,1,2,3 $RUN_CMD

echo "Running: $EVA_CMD"
$EVA_CMD