from builtins import FileNotFoundError
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import faiss
import json
import os 

# 将gsm8k或者其他数据集构建处理成向量库
class DatasetCons:
    def __init__(self, dataset_name, data_path, ds_cot=True, embedding_model_name="GanymedeNil/text2vec-large-chinese"):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.ds_cot = ds_cot   # 这个参数主要是针对gsm8k和ProntoQA数据集，这两个数据集有自己的cot，该参数设置为true时，就使用ds接口生成的cot来构建langchian数据集，为false时，使用这两个数据集默认的cot
    
        self.embedding_path = "../llms/text2vec-large-chinese"
        # self.embedding_model_name = embedding_model_name
        self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_path)   # 直接本地加载

    def gsm8k_load_data(self, split="test"):
        "加载gsm8k数据集"
        "split可选 train test"
        print("current loading function is gsm8k_load_data\(\)")
        # data_file = Path(self.data_path) / f"{self.dataset_name}.txt"
        data_file = Path(self.data_path) / f"{split}.jsonl"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} does not exist.")
        documents = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # print(line)
                # print(type(line))
                data_dict = json.loads(line)
                # print(type(data_dict))
                if line:
                    question = data_dict.get("question", "")
                    answer = data_dict.get("answer", "")
                    cot = data_dict.get("cot", "")
                    
                    documents.append(Document(page_content=question, metadata={"answer": answer, "cot": cot}))
        return documents
    
    # 10.29 加载LogicalDeduction数据集
    def logicaldeduction_load_data(self, split="train"):
        "加载LogicalDeduction数据集，可选train 和dev数据集"
        print("current loading function is logicaldeduction_load_data\(\)")
        data_file = Path(self.data_path) / f"LogicalDeduction_{split}_cot.json"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} does not exist.")
        documents = []
        with open(data_file, "r", encoding="utf-8") as f:
            item_list = json.load(f)
            print(type(item_list), len(item_list))
            for item in item_list:
                context = item.get("context", "")
                question = item.get("question", "")
                options = item.get("options", "")
                answer = item.get("answer", "")
                cot = item.get("reasoning_cot", "")
                # full_prompt = f"Context: {context}\nQuestion: {question}\nExplanation: {explanation}"
                full_prompt = f"Context: {context}\nQuestion: {question}\n"
                documents.append(Document(page_content=full_prompt, metadata={"answer": answer, "cot": cot, "question": question, "context": context, "options": options}))
        return documents
    
    # 20251119 构建一个逻辑推理任务通用的load_data
    def logical_task_common_cot_load_data(self, split="train"):
        print("current loading function is logical_task_common_cot_load_data\(\)")
        data_file = Path(self.data_path) / f"{self.dataset_name}_{split}_cot.json"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} does not exist.")
        documents = []
        with open(data_file, "r", encoding="utf-8") as f:
            item_list = json.load(f)
            print(type(item_list), len(item_list))
            for item in item_list:
                context = item.get("context", "")
                question = item.get("question", "")
                options = item.get("options", "")
                answer = item.get("answer", "")
                cot = item.get("reasoning_cot", "")
                # full_prompt = f"Context: {context}\nQuestion: {question}\nExplanation: {explanation}"
                full_prompt = f"Context: {context}\nQuestion: {question}\n"
                documents.append(Document(page_content=full_prompt, metadata={"answer": answer, "cot": cot, "question": question, "context": context, "options": options}))
        return documents
    
    # 10.26 加载逻辑语言的数据集（这个数据集的explanation用的是数据本身提供的，而非ds生成cot）
    def prontoqa_load_data(self, split="dev"):
        "加载prontoqa数据集"
        "split可选 dev"
        print("current loading function is prontoqa_load_data\(\)")
        data_file = Path(self.data_path) / f"{split}.json"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} does not exist.")
        documents = []
        with open(data_file, "r") as f:
            item_list = json.load(f)
            print(type(item_list), len(item_list))
            for item in item_list:
                context = item.get("context", "")
                explanation = "\n".join(item.get("explanation", ""))
                question = item.get("question", "")
                answer = item.get("answer", "")
                options = item.get("options", "")
                # full_prompt = f"Context: {context}\nQuestion: {question}\nExplanation: {explanation}"
                full_prompt = f"Context: {context}\nQuestion: {question}\n"
                documents.append(Document(page_content=full_prompt, metadata={"answer": answer, "explanation": explanation, "question": question, "context": context, "options": options}))
        return documents

    def build_vector_store(self, index_path, split="dev"):
        if self.dataset_name not in ["ProntoQA","AR-LSAT","ProofWriter", "LogicalDeduction","FOLIO", "gsm8k"]:
            print(f"the wrong dataset {self.dataset_name} were provided. Ended the program!")
            return
        if self.ds_cot==True and self.dataset_name in ["ProntoQA","AR-LSAT","ProofWriter", "LogicalDeduction","FOLIO"]:
            load_fn = self.logical_task_common_cot_load_data
        elif self.dataset_name in ["gsm8k", "ProntoQA"]:
            load_fn = getattr(self, f"{self.dataset_name}_load_data")
        else:
            raise ValueError(f"Unknown dataset:{self.dataset_name}, unrecognized load type:{self.ds_cot}")
#         try:
#             load_fn = getattr(self, f"{self.dataset_name}_load_data")
#         except AttributeError:
#             raise ValueError(f"Unknown dataset: {self.dataset_name}")
        documents = load_fn(split)
        print("the langchain documents num is :", len(documents))
        # exit()
        save_index_path = Path(index_path) / f"{self.dataset_name}"
        if not os.path.exists(save_index_path):
            os.makedirs(save_index_path)
        index_path = str(save_index_path)
        vector_store = faiss.FAISS.from_documents(documents, self.embedding)
        vector_store.save_local(save_index_path)
        print(f"Vector store saved to {save_index_path}")

# 构建一个利用langchain数据库进行处理的类
class DatasetRetriever:
    def __init__(self, index_path, db_name, embedding_model_name="GanymedeNil/text2vec-large-chinese"):
        self.embedding_path = "../llms/text2vec-large-chinese"
        # self.embedding_model_name = embedding_model_name
        self.db_name = db_name   # 用于进行搜索任务的db名
        self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_path)   # 直接本地加载
        # 载入本地faiss向量数据库
        index_path = Path(index_path) / f"{self.db_name}"
        self.vector_store = faiss.FAISS.load_local(index_path, self.embedding, allow_dangerous_deserialization=True)

    def retrieve(self, query, k=5):
        results = self.vector_store.similarity_search(query, k=k)
        retrieve_list = []
        for i, doc in enumerate(results):
            # print(f"Result {i+1}:")
            # print("Content:", doc.page_content)
            # print("Metadata:", doc.metadata)
            # print("-------------------")
            page_content = doc.page_content
            metadata = doc.metadata
            if self.db_name == "gsm8k":
                context = ""
                question = page_content
                answer = metadata.get("answer", "")
                options = ""
                cot = metadata.get("cot", "")
            # 20251119修改
            elif self.db_name in ["ProntoQA","AR-LSAT","ProofWriter", "LogicalDeduction","FOLIO"]:
                context = metadata.get("context", "")
                question = metadata.get("question", "")
                options = metadata.get("options", "")
                answer = metadata.get("answer", "")
                cot = metadata.get("cot", "")
                if self.db_name == "ProntoQA":
                    cot = metadata.get("cot") or metadata.get("explanation", "")
            else:
                raise ValueError("the retriever method are not supported~!", self.dataset_name)
            # 如果检索的数据库和测试数据库是同一个，去掉和query相同的检索结果
            if question in query and context in query:
                continue
            retrieve_list.append({
                "context": context,
                "question": question,
                "options": options,
                "answer": answer,
                "cot": cot
            })
        return retrieve_list

if __name__ == "__main__":
    # 构建langchain检索向量数据库
    # laska 修改使用逻辑语言来进行icl
    data_path = "../rag_data"
#     dataset_cons = DatasetCons(dataset_name="gsm8k", data_path="../data/gsm8k")
#     dataset_cons.build_vector_store("../rag_db","train")

#     dataset_cons = DatasetCons(dataset_name="ProntoQA", data_path="../data/ProntoQA")
#     dataset_cons.build_vector_store("../rag_db","dev")
    
#     dataset_cons = DatasetCons(dataset_name="LogicalDeduction", ds_cot=True,data_path=data_path, )
#     dataset_cons.build_vector_store("../rag_db", "dev")
    
#     dataset_cons = DatasetCons(dataset_name="FOLIO", ds_cot=True, data_path=data_path, )
#     dataset_cons.build_vector_store("../rag_db", "train")
      
    dataset_cons = DatasetCons(dataset_name="ProofWriter", ds_cot=True,data_path=data_path)
    dataset_cons.build_vector_store("../rag_db", "train")
    # 利用构建好的检索向量数据库进行实验
    dataset_retriever = DatasetRetriever(index_path="../rag_db", db_name="ProntoQA")
    dataset_retriever = DatasetRetriever(index_path="../rag_db", db_name="ProofWriter")
#     dataset_retriever = DatasetRetriever(index_path="../rag_db", db_name="LogicalDeduction")
#     dataset_retriever = DatasetRetriever(index_path="../rag_db", db_name="FOLIO")
#     dataset_retriever = DatasetRetriever(index_path="../rag_db", db_name="gsm8k")
    context = "Jompuses are not shy. Jompuses are yumpuses. Each yumpus is aggressive. Each yumpus is a dumpus. Dumpuses are not wooden. Dumpuses are wumpuses. Wumpuses are red. Every wumpus is an impus. Each impus is opaque. Impuses are tumpuses. Numpuses are sour. Tumpuses are not sour. Tumpuses are vumpuses. Vumpuses are earthy. Every vumpus is a zumpus. Zumpuses are small. Zumpuses are rompuses. Max is a yumpus."
    question = "Is the following statement true or false? Max is sour."
    query = f"Context: {context}\nQuestion: {question}\n"
    results = dataset_retriever.retrieve(query, k=3)
    print(results)
    print(len(results))

