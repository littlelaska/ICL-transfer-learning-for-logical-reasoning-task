import json
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArguementParser()
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_name", type=str, default="LogicalDeduction")
    parser.add_argument("--db_name", type=str, help="source dataset")
    parser.add_argument("--db_type", type=str, default="embedding")  # bm25/embedding/cone
    

if __name__ == "__main__":
    cur_dir = Path.cwd()
    log_dir = cur_dir / "logs"
    model_name = "qwen7"
    
    print(cur_dir)