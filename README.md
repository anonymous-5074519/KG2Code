# KG2Code: From Knowledge Graphs to Code for Enhancing Large Language Models

> **Abstract**
Recent research has explored integrating knowledge graphs (KGs) with large language models (LLMs) to enhance their performance on downstream tasks. Despite notable progress, existing methods face key limitations. Most approaches convert KGs into plain text, which fails to preserve their inherent structural information. Others employ adapter modules such as graph neural networks (GNNs) to encode KGs as soft prompts, but these differ from the LLMs’ pre-training paradigm and exhibit poor generalization. To address these issues, we propose KG2Code, a novel framework that transforms knowledge graphs into Python code, a format that LLMs naturally understand and that effectively preserves topological structure. Building on this representation, we design a code-style prompting scheme that reformulates diverse KG-related tasks as the unified code generation problem. Furthermore, we construct a large-scale code-style corpus based on this representation and continue training LLMs to enhance their reasoning and prompt-following abilities. Extensive experiments show that KG2Code substantially outperforms existing KG-enhancement methods on knowledge-intensive tasks, particularly in transfer settings. Our findings demonstrate that code serves as a more expressive and unified medium for integrating structured knowledge, opening a new direction for enhancing LLMs through code-based representation learning.
> 
![](./figs/1.png)
This is the accompanying code for the paper **KG2Code: From Knowledge Graphs to Code for Enhancing Large Language Models**. 

## Setup
### Environment Setup
```
conda create -n CoTKR python=3.12
conda activate CoTKR
pip install -r requirement1.txt
conda create -n tune python=3.12
conda activate tune
pip install -r requirement2.txt
```
**In this work, we use three different environments. For fine-tuning LLMs, please use the tune environment. For the GNN and GNN+Text baselines, please use the GNN environment. For all other cases, please use the KG2Code environment.**
### LLM Setup
We utlize various LLMs in the experiments, including [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct), [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B). Download LLMs to ```pretrain/```.
```
KG2Code/
pretrain/
    ├── Llama-3.1-8B-Instruct/
    ├── DeepSeek-Coder-V2-Lite-Instruct/
    └── DeepSeek-R1-Distill-Llama-8B/                       
```
### Dataset Setup
We evaluate our method on six datasets, [QALD-9-plus](https://github.com/KGQA/QALD_9_plus), [QALD-10](https://github.com/KGQA/QALD-10), [Wikidata5M-Transductive](https://deepgraphlearning.github.io/project/wikidata5m), [Wikidata5M-Transductive](https://deepgraphlearning.github.io/project/wikidata5m), [GrailQA](https://dki-lab.github.io/GrailQA/), and [WN18RR](https://github.com/TimDettmers/ConvE). Download QALD-9-plus to ```KG2Code/inference/KGQA/dataset/QALD-9```. Download QALD-10 to ```KG2Code/inference/KGQA/dataset/QALD-10```. Download Wikidata5M to ```KG2Code/inference/KGC/wikidata5m```. For GrailQA, please download the original [dev set](https://dki-lab.github.io/GrailQA/) to ```KG2Code/inference/KGQA/transfer-KGQA/dataset/grailqa``` and the [2-Hop retrieval results](https://github.com/wuyike2000/CoTKR/tree/main/inference/open/retrieve/2hop/format/grailqa.json) to  ```KG2Code/inference/KGQA/transfer-KGQA/dataset/grailqa```. For WN18RR, please download the [processed data](https://github.com/yao8839836/kg-bert/tree/master/data/WN18RR) to ```KG2Code/inference/KGC/transfer-KGC/dataset/WN18RR```.
```
KG2Code/
└── inference/
    ├── KGQA/
        ├── dataset/
            ├── QALD-9/
                └── qald_9_plus_test_wikidata.json
            └── QALD-10/
                └── qald_10.json
        └── transfer-KGQA/
            └── dataset/
                └── grailqa/
                    ├── grailqa_v1.0_dev.json
                    └── grailqa.json
    └── KGC/
        ├── wikidata5m/
            ├── wikidata5m_all_triplet.txt
            ├── wikidata5m_transductive/
                └── wikidata5m_transductive_test.txt
            └── wikidata5m_inductive/
                └── wikidata5m_inductive_test.txt
        └── transfer-KGC/
            └── dataset/
                └── WN18RR/
                    ├── wordnet-mlj12-definitions.txt
                    ├── train.tsv
                    ├── test.tsv
                    ├── relations.txt
                    ├── relation2text.txt
                    ├── entity2text.txt
                    ├── entities.txt
                    └── dev.tsv
```
## Continual Training
**We provide our fine-tuned KG-Coders lora checkpoints (fine-tuned on code-style corpus) in [KG-Coder.zip]() and our Text model lora checkpoints (fine-tuned on textual corpus) in [Text-Model.zip](). You can download it directly to the following folder and escape the continual training phase. You only need to merge the lora checkpoints into base model.**
### Corpus Construction
**We provide our constructed code-style corpus in [pretrain.zip]() and corresponding textual corpus in [pretrain-text.zip](). You can download it directly to the following folder and escape the following corpus construction steps.**
If you want to generate your corpus, please follow these steps and be ready to spend a lot of money ;)
#### Dict Setup
1. We provide our collected ```endict.pkl``` and ```redict.pkl``` in [dict.zip](). Please download them into ```KG2Code/pretrain/kgqa``` and ```KG2Code/pretrain/kgc```.
2. We provide the ```entity.txt``` and ```relation.txt``` we used in [data.zip](). Please download them into ```KG2Code/pretrain/kgqa``` and ```KG2Code/pretrain/kgc```.
#### KGQA
1. Go to ```KG2Code/pretrain/kgqa```.
2. Run ```dict_filter.py``` to construct the dict files for the following steps.
3. Run ```subgraph.py``` to generate subgraphs used for training corpus generation.
4. Run ```fact_sparql.py``` to generate sparql for factual questions. Run ```count_sparql.py``` to generate sparql for counting questions. Run ```judge_sparql.py``` to generate sparql for boolean questions.
5. Run ```fact_question.py``` to generate questions from factual sparql. Run ```count_question.py``` to generate questions from counting questions. Run ```judge_question.py``` to generate questions from boolean questions. Run ```merge.py``` to merge all these questions into one file.
6. Run ```cot_multi.py``` to generate the Chain-of-Thought (CoT) reasoning process for these questions.
7. Run ```graph_extend.py``` to extend the groundtruth subgraph of the questions.
8. Run ```corpus.py``` to generate the KGQA training corpus for KG2Code. Run ```corpus_text.py``` to generate the KGQA training corpus for Text baseline.
#### KGC
1. Go to ```KG2Code/pretrain/kgc```.
2. Run ```dict.py``` and ```dict-en-re.py``` to construct the dict files for the following steps.
3. Run ```data_multi.py``` to construct the kgc data used for training corpus generation.
4. Run ```cot_multi.py``` to generate the Chain-of-Thought (CoT) reasoning process for these questions.
5. Run ```corpus.py``` to generate the KGC training corpus for KG2Code. Run ```corpus_text.py``` to generate the KGC training corpus for Text baseline.

Go to ```KG2Code/instruction-tuning```. Run ```merge.py``` to generate training corpus for KG2Code and run ```merge_text.py``` to generate training corpus for Text baseline.
### Instruction-tuning
1. Go to ```KG2Code/instruction-tuning```. Run ```train.sh``` to train LLM. Please modify some key parameters like "llm" and "dataset".
2. After training, please put your lora checkpoints into ```KG2Code/inference/checkpoint-code``` for KG2Code and ```KG2Code/inferece/checkpoint-text``` for Text baseline. For example, please put lora checkpoint for KG2Code based on ```Llama-3.1-8B-Instruct``` into ```KG2Code/inference/checkpoint-code/Llama-3.1-8B-Instruct```.
3. Go to ```KG2Code/inference```. Run ```merge.py``` to merge the lora checkpoint into base LLM.
## Inference
### KGQA
1. Go to ```KG2Code/inference/KGQA/retrieve```.
2. Run ```qald-retrieve.py``` to parse sparql and extract the groundtruth subgraph.
3. Run ```graph-extend-stable.py``` to extend the groundtruth subgraph.
4. Run ```graph-query.py``` to query the names of the entities and relations in subgraph. Please modify some key parameters like . ```entity.pkl``` and ```relation.pkl``` are provided in [dict.zip](). Please download them directly.
5. Run ```graph-infer.py``` to change the prompt format into code-style.
6. Go to ```KG2Code/inference/KGQA/answer```.
7. Run ```answer-code.py``` to get the results for KG2Code. Run ```answer-text.py``` to get the results for Text baseline. Run ```answer-origin.py``` to get the results for Raw baseline for Llama-3.1-8B-Instruct. Run ```answer-code-direct.py``` to get the results for Raw baseline for DeepSeek-Coder-V2-Lite-Instruct. Run ```answer-r1.py``` to get the results for R1 baseline.
### Transfer-KGQA
We conduct transfer experiments on GrailQA. For retrieval results, we directly use 2-Hop retrieval results from [CoTKR](https://github.com/wuyike2000/CoTKR).
1. Go to ```KG2Code/inference/KGQA/transfer-KGQA```.
2. Run ```preprocess.py``` to process the files.
3. Go to ```KG2Code/inference/KGQA/transfer-KGQA/answer```.
4. Run ```answer-code.py``` to get the results for KG2Code. Run ```answer-origin.py``` to get the results for Raw baseline. Run ```answer-text.py``` to get the results for Text baseline.
### KGC
1. Go to ```KG2Code/inference/KGC/retrieve```. ```entity.pkl``` and ```relation.pkl``` are provided in [dict.zip](). Please download them directly and put it in ```KG2Code/inference/KGC/retrieve``` and ```KG2Code/inference/KGC/infer```. 
2. Run ```create_dict.py``` to create the dict files for the following steps.
3. Run ```retrieve-2hop.py``` to retrieve the relevant the subgraphs for KGC tasks.
4. Go to ```KG2Code/inference/KGC/infer```.
5. Run ```infer-code.py``` to get the results for KG2Code. Run ```infer-text.py``` to get the results for Text baseline. Run ```infer-origin.py``` to get the results for Raw baseline for Llama-3.1-8B-Instruct. Run ```answer-code-direct.py``` to get the results for Raw baseline for DeepSeek-Coder-V2-Lite-Instruct.
### Transfer-KGC
We conduct transfer experiments on WN18RR. We follow [KG-BERT](https://github.com/yao8839836/kg-bert/tree/master/data/WN18RR) and use the files directly from it.
1. Go to ```KG2Code/inference/KGC/transfer-KGC```.
2. Run ```create_dict.py``` to construct the dict files for the following steps.
3. Run ```retrieve-2hop.py``` to retrieve the subgraph for inference.
4. Go to ```KG2Code/inference/KGC/transfer-KGC/infer```.
5. Run ```infer-code.py``` to get the results for KG2Code. Run ```infer-text.py``` to get the results for Text baseline. Run ```infer-origin.py``` to get the results for Raw baseline.
