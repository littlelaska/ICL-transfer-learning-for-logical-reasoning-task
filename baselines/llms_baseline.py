from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from tqdm import tqdm
import argparse

# 尝试使用vllm加速模型推理
from vllm import LLM, SamplingParams
import torch
from dataset_cons import DatasetRetriever

import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class LLM_Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.demonstration_path = args.demonstration_path
        self.mode = args.mode   # Direct /CoT /RAG
        
        self.all_data_switch = args.all_data_switch  # 是否对完整数据集进行测试
        self.batch_test = args.batch_test  # 是否进行batch测试
        self.batch_size = args.batch_size  # batch size大小
        self.vllm_switch = args.use_vllm  # 是否使用vllm进行加速
        self.max_new_tokens = args.max_new_tokens
        self.zero_shot = args.zero_shot   # 在非rag的模式下生效
        self.rag_result_path = args.rag_result_path
        self.system_prompt_dir = args.system_prompt_dir
        self.user_template_dir = args.user_template_dir
        
        self.dtype = args.dtype
        # 对需要的部分数据进行初始化
        self.para_init()

        self.tokenizer, self.model= self.load_model()
        if not self.vllm_switch:
            self.device = self.model.device

        self.label_phrase = 'The correct option is:'

    # 2025.11.11 separate some init code from init
    def para_init(self):
        # 模型路径初始化
        if self.model_name == "qwen7":
            self.model_path = "../llms/Qwen2.5-7B-Instruct"
        elif self.model_name == "qwen14":
            self.model_path = "../llms/Qwen2.5-14B-Instruct"
        elif self.model_name == "qwen3-8":
            self.model_path = "../llms/Qwen3-8B"
        elif self.model_name == "qwen3-14":
            self.model_path = "../llms/Qwen3-14B"
        elif self.model_name == "qwen3-32":
            self.model_path = "../llms/Qwen3-32B"
        elif self.model_name == "llama3-8":
            self.model_path = "../llms/llama3.1-8B-Instruct"
        else:
            self.model_path = "../llms/"
        
        # 针对不同mode的参数初始化
        if self.mode == "RAG":   # 说明当前是rag模式，需要加载检索库
            # RAG检索器加载部分
            self.rag_topk = args.top_k   # 检索的样例个数
            self.rag_icl_num = args.icl_num   # 用于上下文学习的展示样例个数  
            self.db_name = args.db_name 
            self.index_path = args.index_path
            self.dataset_retriever = DatasetRetriever(self.args)
            self.db_type = args.db_type
            # rag所用的icl template文件路径，用于包装检索到的document
            self.icl_template_file =  f"{'gsm8k' if self.db_name == 'gsm8k' else 'LogicalReasoning'}_ICL_template.txt"
            self.icl_template_path = os.path.join(self.user_template_dir, self.icl_template_file)
        # 将zero-shot的逻辑也加到这里
        elif self.zero_shot:
            self.testing_type = "0-shot"
        else:
            self.testing_type = "few-shot"
        
        # role prompt 路径初始化
        if self.dataset_name == "gsm8k":
            self.prompt_file = f"{self.dataset_name}_{self.mode}{'_0shot' if self.zero_shot else ''}.txt"
        else:
            self.prompt_file = f"LogicalReasoning_{self.mode}{'_0shot' if self.zero_shot else ''}.txt"
        self.system_prompt_path = os.path.join(self.system_prompt_dir, self.prompt_file)
        print(f"system prompt file path: {self.system_prompt_path}")
        
        # user prompt路径初始化
        self.user_prompt_path = os.path.join(self.user_template_dir, self.prompt_file)
        print(f"user prompt file path: {self.user_prompt_path}")
        
        # 待检查判断逻辑是否正确及完善， prompt creator初始化
        if self.mode == "RAG":
            if self.rag_icl_num > 0:
#                 self.prompt_creator = self.rag_prompt_creator
                self.prompt_creator = self.prompt_LSAT
            else:
                self.prompt_creator = self.prompt_LSAT
        else:
            self.prompt_creator = self.prompt_LSAT

        # 结果存储路径初始化
        # 统一定义存储路径
        if self.mode == "RAG":
            self.save_file = os.path.join(self.save_path, f'{self.mode}{self.rag_icl_num}_{self.db_name}_{self.db_type}_{self.dataset_name}_{self.split}_{self.model_name}.json')
            # laska定义一个保存检索中间结果的文件
            if not os.path.exists(self.rag_result_path):
                os.makedirs(self.rag_result_path)
            self.retrieval_save_file = os.path.join(self.rag_result_path, f'retrieval_{self.db_name}_{self.db_type}_{self.dataset_name}_{self.split}.json')   # 只与文件有关
            self.retrieval_writer = open(self.retrieval_save_file, 'w') 
        else:
            self.save_file = os.path.join(self.save_path, f'{self.mode}_{self.testing_type}_{self.dataset_name}_{self.split}_{self.model_name}.json')
        
        # 打印部分参数
        print("="*16+"parameteres"+"="*16)
    
        self.print_self()
        print("="*16+"parameteres"+"="*16)
    # 打印参数
    def print_self(self):
        for k,v in self.__dict__.items():
            print(f"{k}:{v}")

    # laska system prompt加载函数
    def load_system_prompt(self):
        with open(self.system_prompt_path, 'r') as f:
            system_prompt = f.read()
        return system_prompt

    # 2025.11.11 加载user prompt的template部分，用于构建数据
    def load_user_prompt_template(self):
        with open(self.user_prompt_path, "r") as f:
            user_prompt = f.read()
        return user_prompt
    
    # 2025.11.11 增加icl_prompt部分
    def load_icl_template(self):
        with open(self.icl_template_path, "r") as f:
            icl_template = f.read()
        return icl_template
    
    # laska 模型加载部分     
    def load_model(self):
        # vllm 新增
        if self.vllm_switch:
            print("使用vllm进行模型加载和推理")
            print("loading model from:", self.model_path)
            model = LLM(model=self.model_path, tokenizer=self.model_path,tensor_parallel_size=torch.cuda.device_count(), max_model_len=32768,dtype=self.dtype, trust_remote_code=True, gpu_memory_utilization=0.9)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left')
#             self.sampling_params = SamplingParams(temperature=0, max_tokens=self.max_new_tokens, top_p=0.95, top_k=40, n=1)
            self.sampling_params = SamplingParams(temperature=0, max_tokens=self.max_new_tokens, top_p=1, top_k=1, n=1)
            return tokenizer, model
        else:
            print("直接加载模型进行推理")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left')   # 直接从本地路径进行加载
            print("loading model from:", self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype="auto", device_map="auto")
            print("loading complete")
            return tokenizer, model

    # laska 构建使用rag动态变化demonstration的prompt生成器
    def rag_prompt_creator(self, in_context_example, test_example):
        # 2025.11.11 add system prompt
        role_content = self.load_system_prompt()   # 不论是rag还是cot的system prompt都是一样的
        user_prompt_template = self.load_user_prompt_template()  # 目前这一部分的选择是不一样的
        # 首先进行检索，得到相关的demonstration
        # 所有数据集都有question域
        rag_query =test_example["question"].strip()
        retrieved_results = self.dataset_retriever.retrieve(rag_query, self.rag_topk)
        # 制定一个template 
        icl_template = self.load_icl_template()
        print(icl_template)
#         icl_template = "Context:\n{context}\nQuestion:\n{question}\nOptions:\n{options}\nReasoning:\n{cot}\nAnswer:\n{answer}\n"
        
        # 构建检索的数据集
        overall_demonstration = ""
        for result in retrieved_results[:self.rag_icl_num]:
            overall_demonstration += icl_template.format(
                context=result['context'],
                question=result['question'],
                options='\n'.join([opt.strip() for opt in result.get("options", [])]),
                cot=result['cot'],
                answer=result['answer']
            ) + "\n"
        
        full_in_context_example = user_prompt_template.replace("[[DEMONSTRATIONS]]",)
#         full_in_context_example = head_template + "\n" + overall_demonstration
        # 将需要测试的内容进行拼接
        test_template = "Context:\n{context}\nQuestion:\n{question}\nOptions:\n{options}\nReasoning:"
#         print(test_example)
#         print(test_example["context"])
        test_example_str = test_template.format(context=test_example['context'],
                                                question=test_example['question'],
                                                options='\n'.join([opt.strip() for opt in test_example['options']]))
        # 拼接成为最终给模型进行测试的样例
        full_prompt = full_in_context_example + "\n" + test_example_str
        role_content = "You are a logical task solver. Follow the demonstrationa to solve the new question. Remember to think step by step with concise chain-of-thought, and adhere to the context related to the question. Then on a new line, output exactly: 'The correct option is: A' or 'The correct option is: B"
        messages = [
            {"role":"system", "content":role_content},
            {"role":"user", "content": full_prompt}
            ]
        print(messages)
        # laska 修改，针对本地模型，返回messages
        # 每检索一条，将检索结果写入文件
        retrieval_record = {
            'context': test_example['context'],
            'question': test_example['question'],
            'retrieved_demonstrations': full_in_context_example
        }
        # 写入json文件
        self.retrieval_writer.write(json.dumps(retrieval_record, ensure_ascii=False) + '\n')
        return messages
    
    # 针对few-shot，生成prompt，该部分完成的是在单个样例之前添加few-shot的示例
    def prompt_LSAT(self, in_context_example, test_example):
        # 2025.11.11 add system prompt
        role_content = self.load_system_prompt()   # 不论是rag还是cot的system prompt都是一样的
        user_prompt_template = self.load_user_prompt_template()  # 目前这一部分的选择是不一样的     
        # 这一部分分支逻辑待验证代码正确性
        if self.mode == "RAG":
            full_prompt = user_prompt_template
        elif self.zero_shot == True:
            full_prompt = user_prompt_template
        else:
            full_prompt = in_context_example
        
        # 2025.11.11 增加rag的prompt构造
        # 所有数据集都有question域
        if self.mode == "RAG":
            rag_query = test_example["question"].strip()
            retrieved_results = self.dataset_retriever.retrieve(rag_query, self.rag_topk)
            # 制定一个template 
            icl_template = self.load_icl_template()
#             print(icl_template)
            # 构建检索的数据集
            overall_demonstration = ""
            for result in retrieved_results[:self.rag_icl_num]:
                overall_demonstration += icl_template.format(
                    context=result['context'],
                    question=result['question'],
                    options='\n'.join([opt.strip() for opt in result.get("options", [])]),
                    cot=result['cot'],
                    answer=result['answer']
                ) + "\n"
#             print("before replace:\n", overall_demonstration)
            full_prompt = user_prompt_template.replace("[[DEMONSTRATIONS]]", overall_demonstration)
        
        # 针对role paly的模型，需要加上user等角色
        # 针对gsm8k的处理逻辑不一样
        if self.dataset_name == "gsm8k":
            question = test_example['question'].strip()
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
        else:
            context = test_example['context'].strip()
            question = test_example['question'].strip()
            options = '\n'.join([opt.strip() for opt in test_example['options']])
            full_prompt = full_prompt.replace('[[CONTEXT]]', context)
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
            full_prompt = full_prompt.replace('[[OPTIONS]]', options)
        messages = [
            {"role":"system", "content":role_content},
            {"role":"user", "content": full_prompt}
            ]
        if self.mode == "RAG":
            # laska 修改，针对本地模型，返回messages
            # 每检索一条，将检索结果写入文件
            retrieval_record = {
                'context': test_example.get('context',""),
                'question': test_example['question'],
                'retrieved_demonstrations': full_prompt
            }
            # 写入json文件
            self.retrieval_writer.write(json.dumps(retrieval_record, ensure_ascii=False) + '\n')
        return messages

    # 针对zero-shot，直接生成prompt
    def prompt_LSAT_zero_shot(self, in_context_example, test_example):
        # 2025.11.11 add system prompt
        role_content = self.load_system_prompt()
        user_prompt_template = self.load_user_prompt_template()  # 目前这一部分的选择是不一样的
        full_prompt = user_prompt_template
        # 针对gsm8k的处理逻辑不一样
        if self.dataset_name == "gsm8k":
            question = test_example['question'].strip()
            # full_prompt = f"Problem: {question}\nReasoning:"
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
        else:  # 针对其他逻辑推理的数据集
            context = test_example['context'].strip()
            question = test_example['question'].strip()
            options = '\n'.join([opt.strip() for opt in test_example['options']])
            full_prompt = full_prompt.replace('[[CONTEXT]]', context)
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
            full_prompt = full_prompt.replace('[[OPTIONS]]', options)
            # # laska 10.27测试逻辑prompt
            # if self.mode == 'Logical':
            #     full_prompt = f"Context: {context}\nQuestion: {question}\n"
            # elif self.mode == "Direct":
            #     full_prompt = f"Context: {context}\nQuestion: {question}\nOptions:\n{options}\nPlease answer the question directly, directly give the answer option. The correct option is:"
            # elif self.mode == "CoT":
          
            #     full_prompt = f"Context: {context}\nQuestion: {question}\nOptions:\n{options}\nLet's think step by step. The correct option is:"

        messages = [
            {"role":"system", "content":role_content},  
            {"role":"user", "content": full_prompt}
            ]   
        return messages
       
    # 针对few-shot的处理代码
    def load_in_context_examples(self):
        with open(os.path.join(self.demonstration_path, f'{self.dataset_name}_{self.mode}.txt')) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_raw_dataset(self, split):
        if self.dataset_name == "gsm8k":
            with open(os.path.join(self.data_path, self.dataset_name, f"{self.split}.jsonl"), 'r') as f:
                raw_dataset = [json.loads(line) for line in f]
            return raw_dataset
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    # laska 调用本地模型生成结果
    def model_generate(self, messages):
        # print(type(messages), type(messages[0]), len(messages), len(messages[0]))
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # apply之后得到的text是一个字符串，而tokenizer的输入需要是一个list，所以需要[text]
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**model_inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        # model.generate返回的结果是一个[[... ...]]的二维list，单条和batch的区别在于第一维的长度
        # print("--------the final answer is !!!!---------")
        # print(generated_ids)
        # 针对单条数据，需要去掉前面input_id的部分
        generated_ids = generated_ids[:,len(model_inputs.input_ids[0]):]
        # print(generated_ids.shape)
        # print(generated_ids)
        # response的返回是列表的形式，针对单条数据的测试，需要取第1条元素
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print("++++++the response is ++++++++++")
        # print(response)
        return response
    
    # laska 定义一个针对batch数据进行解码的函数
    def model_generate_batch(self, messages_list):
        texts = [self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]
        # texts 经过apply函数之后是str的列表
        # print(type(texts), len(texts), type(texts[0]), len(texts[0]))
        # 分为vllm和普通调用两部分
        if self.vllm_switch:
            # vllm的调用，与model generate不同，不需要进行tokenizer的encode
            outputs = self.model.generate(texts, sampling_params=self.sampling_params)
            # print(outputs)
            responses = [output.outputs[0].text for output in outputs]
            # print(responses)
                
        else:
            model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            generated_ids = self.model.generate(**model_inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
            # print("--------the final answer is !!!!---------")
            # print(generated_ids)
            generated_ids = [output_ids[len(input_ids):] for output_ids, input_ids in zip(generated_ids, model_inputs.input_ids)]
            # response
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # print(responses)
        return responses  # 返回当前一个batch的结果
    
    # laska 定义一个调用入口, 分配是batch还是单条
    def generation_entrance(self):
        if self.batch_test:
            print("进行batch测试")
            self.batch_reasoning_graph_generation(batch_size=self.batch_size)
        else:
            print("进行单条测试")
            self.reasoning_graph_generation()

    def reasoning_graph_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples
        in_context_examples = self.load_in_context_examples()
        
        outputs = []
        for example in tqdm(raw_dataset):
            question = example['question']

            # create prompt
            full_prompt = self.prompt_creator(in_context_examples, example)
#             print(full_prompt)
            # 修改这部分模型生成代码
#             output = self.openai_api.generate(full_prompt)
            # output = self.model_generate(full_prompt)
            # laska ，修改为同一个函数调用，唯一的差别是list中的元素个数
            # 此处的full_prompt是一个list，qwen的输入格式，包含system和user两个部分
            outputs = self.model_generate_batch(full_prompt)
            output = outputs[0]  # 取出单条数据的结果
            # get the answer
            label_phrase = self.label_phrase    #  self.label_phrase = 'The correct option is:'
            generated_answer = output.split(label_phrase)[-1].strip()
            generated_reasoning = output.split(label_phrase)[0].strip()

            # create output
            output = {'id': example['id'], 
                      'question': question, 
                      'answer': example['answer'], 
                      'predicted_reasoning': generated_reasoning,
                      'predicted_answer': generated_answer}
            outputs.append(output)
            # 定义一个测试的开关
            if self.all_data_switch == False:
                print(full_prompt)
                print("当前只测试一条数据，查看结果即可")
                print(output)
                break
        # save outputs        
        with open(self.save_file, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    # laska 定义一个batch测试的代码
    def batch_reasoning_graph_generation(self, batch_size=10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples,针对非0-shot的场景
        if self.mode in ["CoT", "Direct"] and not self.zero_shot:    # rag形式需要自行查找context
            in_context_examples = self.load_in_context_examples()
        else:
            in_context_examples = ""
            
        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            # create prompt
            full_prompts = [self.prompt_creator(in_context_examples, example) for example in chunk]
            # 调用模型进行batch的预测
            batch_output = self.model_generate_batch(full_prompts)
            for sample, output in zip(chunk, batch_output):
                # get the answer
                dict_output = self.update_answer(sample, output)
                outputs.append(dict_output)
            # 定义一个测试的开关
            if self.all_data_switch == False:
                print(full_prompts)
                print("当前只测试一个batch数据，查看结果即可")
                print(outputs)
                break
        # save outputs        
        with open(self.save_file, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def update_answer(self, sample, output):
        # 针对gsm8k是单独的处理
        if self.dataset_name == "gsm8k":
            label_phrase = "Final answer:"
            generated_answer = output.split(label_phrase)[-1].strip().lstrip("<").rstrip(">")
            generated_reasoning = output.split(label_phrase)[0].strip()
        # 针对其他逻辑推理的数据集ProntoQA、ProofWriter等
        else:    
            if self.mode in ["Direct", "CoT", "RAG"]:
                label_phrase = self.label_phrase
            elif self.mode in ["Logical"]:
                label_phrase = "Answer:"
                
            if label_phrase not in output and label_phrase.lower() in output:
                label_phrase = label_phrase.lower()
            generated_answer = output.split(label_phrase)[-1].strip()
            if generated_answer.lower() == "true":
                generated_answer = "A"
            elif generated_answer.lower() == "false":
                generated_answer = "B"
            generated_reasoning = output.split(label_phrase)[0].strip()
        dict_output = {'id': sample['id'], 
                        'question': sample['question'], 
                        'answer': sample['answer'], 
                        'predicted_reasoning': generated_reasoning,
                        'predicted_answer': generated_answer,
                        'generation_context':output}
        return dict_output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
#     parser.add_argument('--model_path', type=str, default='../llms')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str, help='Direct or CoT or logical', default='Direct')
    parser.add_argument('--max_new_tokens', type=int)
    # laska定义一个针对0-shot的代码
    parser.add_argument('--zero_shot', default=False, action='store_true')
    # laska 定义一个batch测试的开关
    parser.add_argument('--batch_test', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    # 定义一个vllm的开关
    parser.add_argument('--use_vllm', default=False, action='store_true')
    # laska 定义一个针对是否对完整数据集进行测试的开关
    parser.add_argument('--all_data_switch', help='当前是否需要对所有数据集进行测试(True)，还是测试代码功能(Fasle:只测试一条数据就可以)', default=False, action='store_true')
    # 10.27 将system prompt放在文件中进行加载
    parser.add_argument('--system_prompt_dir', type=str, default='./system_prompt', help="定义存放system prompt的文件路径")
    # parser.add_argument('--prompt_file', help="定义system prompt的文件路径", type=str, default='logical_prompt_1.txt')
    # 11.7 将rag功能直接加进来
    parser.add_argument('--db_name', type=str, default='gsm8k', help="所使用的RAG db的名字")  # 用于检索的数据库名称
    parser.add_argument('--index_path', type=str, default='../rag_db', help="RAG向量数据库的路径")  # RAG向量数据库的路径
    parser.add_argument('--icl_num', type=int, default=0, help="RAG检索后使用的示例个数")  # RAG检索后使用的示例个数
    parser.add_argument('--top_k', type=int, default=3, help="RAG检索的top k个数")  # RAG检索的top k个数
    parser.add_argument('--rag_result_path', type=str, default='./rag_results', help="RAG检索中间结果的保存路径")  # RAG检索中间结果的保存路径
    parser.add_argument("--db_type", type=str, help="可选的langchain db类型，embedding或者bm25", default="embedding")
    # 2025.11.11 user_template_dir
    parser.add_argument("--user_template_dir", type=str, default="./user_template", help="用于存放user template文件的dir路径")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    llm_problem_reduction = LLM_Reasoning_Graph_Baseline(args)
    # 尝试全部直接调用batch的生成代码
    llm_problem_reduction.generation_entrance()
#     llm_problem_reduction.batch_reasoning_graph_generation(batch_size=10)
    # llm_problem_reduction.reasoning_graph_generation()
