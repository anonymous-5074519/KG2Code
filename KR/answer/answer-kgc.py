import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # force GPU id=0 for vLLM

import json
from tqdm import tqdm
import torch
import sys
import re
import pickle
from pathlib import Path

from vllm import LLM, SamplingParams

# -----------------------------
# Basic configurations (kept from the original script)
# -----------------------------
KR = 'kg-to-text'                  # 'kg-to-text' | 'summary' | others (with reasoning chain)
DATASET = 'wikidata5m_transductive'
REWRITE = 'Meta-Llama-3.1-8B-Instruct'
# LLM_NAME: Meta-Llama-3.1-8B-Instruct, DeepSeek-Coder-V2-Lite-Instruct
LLM_NAME = 'DeepSeek-Coder-V2-Lite-Instruct'

# Explicit batch size for vLLM generation
BATCH_SIZE = 64

# I/O paths (same structure as original)
test = json.load(open(f'../rewrite/{DATASET}/{REWRITE}/{KR}.json', 'r', encoding='utf-8'))
result_path = f'{DATASET}/{REWRITE}/{LLM_NAME}/{KR}.json'
os.makedirs(f'{DATASET}/{REWRITE}/{LLM_NAME}', exist_ok=True)

# Local model path for vLLM
LLM_PATH = f'../../../pretrain/{LLM_NAME}'

# -----------------------------
# Load cached resources
# -----------------------------
# endict: entity_id -> {'label': str, ...}
# out_en_re / in_en_re: adjacency maps for (head, relation) and (tail, relation)
endict = pickle.load(open('endict.pkl', 'rb'))
out_en_re = pickle.load(open("out_en_re.pkl", "rb"))
in_en_re = pickle.load(open("in_en_re.pkl", "rb"))

# -----------------------------
# Prompt template (kept from the original script)
# -----------------------------
base_prompt = '''Please infer the missing part of the triple based on the following knowledge. First, provide your Chain-of-Thought (CoT) reasoning process. At the end, list all plausible answers in a single line, starting with "[MASK]: " and separating each answer by "|" as follows: [MASK]: First answer|Second answer|Third answer ...
{knowledge}
Triple: {triple}
'''

# -----------------------------
# Metrics
# -----------------------------
def cal_metric(predict, answer, gt):
    """Compute raw and filtered Hits@k metrics."""
    pred = [str(w).lower() for w in predict]
    ans_set = {str(w).lower() for w in answer}
    gt = str(gt).lower()

    raw_1  = int(gt in pred[:1])
    raw_3  = int(gt in pred[:3])
    raw_10 = int(gt in pred[:10])

    # Filter out other gold answers when computing filtered metrics
    others = ans_set - {gt}
    filtered_pred = [p for p in pred if p not in others]

    filtered_1  = int(gt in filtered_pred[:1])
    filtered_3  = int(gt in filtered_pred[:3])
    filtered_10 = int(gt in filtered_pred[:10])

    return {
        "raw": {"hits@1": raw_1, "hits@3": raw_3, "hits@10": raw_10},
        "filtered": {"hits@1": filtered_1, "hits@3": filtered_3, "hits@10": filtered_10},
    }

# -----------------------------
# vLLM engine
# -----------------------------
def build_engine():
    """
    Build the vLLM engine. We rely on llm.get_tokenizer() to retrieve EOS/Pad token ids.
    """
    llm = LLM(
        model=LLM_PATH,
        dtype="half",                 # fp16 for speed/memory
        tensor_parallel_size=1,       # single visible GPU
        trust_remote_code=True,       # allow custom model code if needed
        gpu_memory_utilization=0.9,
        max_model_len=4096
    )
    return llm

# -----------------------------
# vLLM batched generation
# -----------------------------
def llm_response_batch(prompts, llm, sampling_params):
    """
    Generate model responses for a batch of prompts using vLLM.
    Returns a list of decoded strings in the same order as prompts.
    """
    results = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False
    )
    texts = []
    for r in results:
        if len(r.outputs) == 0:
            texts.append("")
        else:
            texts.append(r.outputs[0].text.strip())
    return texts

# -----------------------------
# Utilities: parse predictions from "[MASK]:" line
# -----------------------------
def extract_mask_answers(text):
    """
    Parse the model output and extract candidate answers from a line starting with "[MASK]:".
    Returns a list of unique, lowercased strings in order of appearance.
    """
    preds = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('[MASK]:'):
            parts = line.replace('[MASK]:', '').strip().split('|')
            for a in parts:
                a = a.strip().lower()
                if a and a not in preds:
                    preds.append(a)
            break
    return preds

# -----------------------------
# Main
# -----------------------------
def main():
    # Build vLLM engine and tokenizer
    llm = build_engine()
    tokenizer = llm.get_tokenizer()

    # vLLM sampling configuration (mirrors your original GenerationConfig)
    sampling_params = SamplingParams(
        temperature=0.01,
        top_k=40,
        top_p=0.9,
        n=1,
        max_tokens=1024,
        repetition_penalty=1.1,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    # Running metrics
    index = 0
    hit1_r = hit3_r = hit10_r = 0
    hit1_f = hit3_f = hit10_f = 0
    data = []

    pbar = tqdm(total=len(test))
    for start in range(0, len(test), BATCH_SIZE):
        batch_samples = test[start:start + BATCH_SIZE]

        # Build prompts for this batch
        batch_prompts = [
            base_prompt.format(knowledge=s['knowledge'], triple=s['mask_triple'])
            for s in batch_samples
        ]

        # Inference with vLLM
        batch_responses = llm_response_batch(batch_prompts, llm, sampling_params)

        # Parse responses and compute metrics
        for sample, response in zip(batch_samples, batch_responses):
            # Construct answer candidate set based on link direction
            h, r, t = sample["triple"].split('|')
            if sample["type"] == 'tail':
                ans_candidate = list(out_en_re[(h, r)])
            else:
                ans_candidate = list(in_en_re[(t, r)])

            # Map entity ids -> readable labels
            ans_e, ans_n = [], []
            for a in ans_candidate:
                if endict.get(a) and endict[a]['label'] is not None:
                    ans_e.append(a)
                    ans_n.append(endict[a]['label'])

            # Save raw response
            sample['response'] = response

            # Extract predicted answers from a "[MASK]:" line
            re_answer = extract_mask_answers(response)
            sample['response_answer'] = re_answer

            # Compute raw/filtered Hits@k
            result = cal_metric(re_answer, ans_n, sample["gt_name"])
            sample["result"] = result
            data.append(sample)

            # Accumulate running metrics
            index += 1
            hit1_r += result["raw"]["hits@1"]
            hit3_r += result["raw"]["hits@3"]
            hit10_r += result["raw"]["hits@10"]
            hit1_f += result["filtered"]["hits@1"]
            hit3_f += result["filtered"]["hits@3"]
            hit10_f += result["filtered"]["hits@10"]

            # Streaming logs
            print('*' * 30, 'Raw Metrics', '*' * 30)
            print('Current Hits@1: {}'.format(hit1_r / index))
            print('Current Hits@3: {}'.format(hit3_r / index))
            print('Current Hits@10: {}'.format(hit10_r / index))
            print('*' * 30, 'Filtered Metrics', '*' * 30)
            print('Current Hits@1: {}'.format(hit1_f / index))
            print('Current Hits@3: {}'.format(hit3_f / index))
            print('Current Hits@10: {}'.format(hit10_f / index))
            sys.stdout.flush()

        pbar.update(len(batch_samples))
    pbar.close()

    # Final metrics
    total = len(test)
    print('*' * 30, 'Final Metrics', '*' * 30)
    print('*' * 30, 'Final Raw Metrics', '*' * 30)
    print('Hits@1: {}'.format(hit1_r / total))
    print('Hits@3: {}'.format(hit3_r / total))
    print('Hits@10: {}'.format(hit10_r / total))
    print('*' * 30, 'Final Filtered Metrics', '*' * 30)
    print('Hits@1: {}'.format(hit1_f / total))
    print('Hits@3: {}'.format(hit3_f / total))
    print('Hits@10: {}'.format(hit10_f / total))

    # Persist results
    json.dump(data, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Disable gradients for safety/speed
    torch.set_grad_enabled(False)
    main()
