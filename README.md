# Cross-domain Knowledge Transfer

Data and Codes for ["Reason Analogically via Cross-domain Prior Knowledge: An Empirical Study of Cross-domain Knowledge Transfer for In-Context Learning"](https://arxiv.org/). 

## Introduction

This repository contains all the code and data used in our study on transfer learning within **in-context learning (ICL) for logical reasoning tasks**. We systematically examine how cross-domain ICL performance varies across **different model families and sizes** (e.g., LLaMA, Gemma, 7B–14B models), as well as across diverse **demonstration retrieval strategies** (e.g., random retireval, BM25, embedding-based similarity search, and retrieval-and-reranking methods) and **different source-target domain configurations**.
Through extensive experiments and detailed analyses, we derive a set of **generalizable insights** that characterize cross-domain reasoning transfer in ICL and offer practical guidance for future research.
For a complete description of our findings, please refer to our paper.

First, install all the required packages:

```bash
pip install -r requirements.txt
```

## Datasets

The datasets we used are preprocessed and stored in the `./data` folder. We evaluate on the following datasets:

- [ProntoQA](https://github.com/asaparov/prontoqa): Deductive resoning dataset. We use the 5-hop subset of the *fictional characters* version, consisting of 500 testing examples. 
- [ProofWriter](https://allenai.org/data/proofwriter): Deductive resoning dataset. We use the depth-5 subset of the OWA version. To reduce overall experimentation costs, we randomly sample 600 examples in the test set and ensure a balanced label distribution.
- [FOLIO](https://github.com/Yale-LILY/FOLIO): First-Order Logic reasoning dataset. We use the entire FOLIO test set for evaluation, consisting of 204 examples.
- [LogicalDeduction](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/logical_deduction): Constraint Satisfaction Problems (CSPs). We use the full test set consisting of 300 examples.
- [AR-LSAT](https://github.com/zhongwanjun/AR-LSAT): Analytical Reasoning (AR) problems, containing all analytical logic reasoning questions from the Law School Admission Test from 1991 to 2016. We use the test set which has 230 multiple-choice questions. 
- [gsm8k](https://github.com/openai/grade-school-math): Grade-school math word problem dataset designed to evaluate multi-step arithmetic reasoning. We use the standard test split consisting of 1,319 examples. Each problem requires synthesizing numerical facts, performing multi-hop computations, and producing a single numerical answer, making GSM8K a strong benchmark for assessing arithmetic reasoning capabilities in LLMs.

## Baselines

To replicate the results show in the paper, please run the following commands:

```bash
cd ./baselines
bash run_yanjzh.sh
```

This [run_batch.sh](xxx) script provides a unified interface for **batch-running cross-domain ICL experiments**. By configuring options inside the script, you can automatically run multiple experiments across, The major parameters and their possible values are as follows:
:

- **MODEL_NAME**: The basic model used in this batch of test. (e.g., *Qwen2.5-7B*, *Gemma3-27B*, etc.)  
- **DB_TYPE**: The retrieve method used in this test, available options include: random, bm25, embedding.
- **EMBEDDING_MODEL**: The model used to encode the query and context into vector embedding, to support further retrieval.
- **SOURCE_DOMAINS**: The source domain used in the transfer.
- **TARGET_DOMAINS**: The target domain which to be benefit from the above source domain.
- **SHOTS**: All the icl demonstration numbers to be tested.
- **CONE_RERANK**: Set to *True* means using the CoNE rerank method (`cone_rerank=True`)

It is designed to streamline large-scale experimentation and ensure consistent, reproducifiable evaluation across all configurations.

The results will be saved in `./baselines/results`. The running logs will be saved in `./baselines/logs`.
