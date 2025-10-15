import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Select GPU

import json
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams

# -----------------------------
# Basic Configurations (same as original script)
# -----------------------------
KR = 'kg-to-text'                  # 'kg-to-text' | 'summary' | others (with reasoning chain)
TASK = 'KGC'                 # 'KGQA' | other tasks (e.g., triple completion)
DATASET = 'wikidata5m_transductive'
LLM_NAME = 'Meta-Llama-3.1-8B-Instruct'
BATCH_SIZE = 64               # batch size, adjustable based on GPU memory

# I/O
data = json.load(open(f'../../inference/{TASK}/graph/{DATASET}/test.json','r',encoding='utf-8'))
result_path = f'{DATASET}/{LLM_NAME}/{KR}.json'
os.makedirs(f'{DATASET}/{LLM_NAME}', exist_ok=True)

# Model and tokenizer path (same as original script)
LLM_PATH = f'../instruction-tuning/output/{KR}/{LLM_NAME}/merge'

# -----------------------------
# Prompt Templates (same as original script)
# -----------------------------
if KR == 'kg-to-text':
    base_prompt = '''Your task is to transform a knowledge graph to a sentence or multiple sentences. The knowledge graph is: {graph}. The sentence is: '''
elif KR == 'summary':
    if TASK == 'KGQA':
        base_prompt = '''Your task is to summarize the relevant knowledge that is helpful to answer the question from the following subgraph.
Subgraph: {graph}
Question: {ques}
Knowledge: '''
    else:
        base_prompt = '''Your task is to summarize the relevant knowledge that is helpful to predict the missing part of the triple from the following subgraph in one line.
Subgraph: {graph}
Triple: {triple}
Knowledge: '''
else:
    if TASK == 'KGQA':
        base_prompt = '''Your task is to summarize the relevant information that is helpful to answer the question from the following subgraph. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Subgraph: {graph}
Question: {ques}
'''
    else:
        base_prompt = '''Your task is to summarize the relevant information that is helpful to predict the missing part of the triple from the following subgraph. Please think step by step and iteratively generate the reasoning chain and the corresponding knowledge.
Subgraph: {graph}
Triple: {triple}
'''

# -----------------------------
# vLLM Engine
# -----------------------------
def build_engine():
    llm = LLM(
        model=LLM_PATH,
        dtype="half",               # fp16 for efficiency
        tensor_parallel_size=1,     # single GPU
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=16384
    )
    return llm

def llm_response_batch(prompts, llm, sampling_params):
    results = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False
    )
    texts = []
    for r in results:
        texts.append(r.outputs[0].text.strip() if r.outputs else "")
    return texts

# -----------------------------
# Main: Batch Inference
# -----------------------------
def main():
    llm = build_engine()
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.01,
        top_k=40,
        top_p=0.9,
        n=1,
        max_tokens=1024,
        repetition_penalty=1.1,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
    )

    # debug
    #global data
    #data=data[:10]
    ###
    processed = []
    pbar = tqdm(total=len(data))
    for start in range(0, len(data), BATCH_SIZE):
        batch_samples = data[start:start + BATCH_SIZE]

        # ---- Build graph_str and prompt input (same logic as original script) ----
        batch_inputs = []
        for sample in batch_samples:
            if TASK == 'KGQA':
                graph_str = ''
                for t in sample["graph_extend_name"]:
                    graph_str = graph_str + '(' + ', '.join(t) + ') '
                graph_str = graph_str[:-1] if graph_str else ''
            else:
                graph_str = ''
                for t in sample["graph_name"]:
                    graph_str = graph_str + '(' + ', '.join(t) + ') '
                graph_str = graph_str[:-1] if graph_str else ''

            if KR == 'kg-to-text':
                inputs = base_prompt.format(graph=graph_str)
            else:
                if TASK == 'KGQA':
                    inputs = base_prompt.format(graph=graph_str, ques=sample["question"])
                else:
                    inputs = base_prompt.format(graph=graph_str, triple=sample["mask_triple"])

            batch_inputs.append(inputs)

        # ---- Run vLLM batch generation ----
        batch_outputs = llm_response_batch(batch_inputs, llm, sampling_params)

        # ---- Write results back to sample, with same field name: sample["knowledge"] ----
        for sample, knowledge in zip(batch_samples, batch_outputs):
            sample_out = dict(sample)  # avoid modifying original object
            sample_out["knowledge"] = knowledge
            processed.append(sample_out)

        pbar.update(len(batch_samples))
    pbar.close()

    # ---- Save results ----
    json.dump(processed, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"Saved to: {result_path}  |  Total samples: {len(processed)}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
