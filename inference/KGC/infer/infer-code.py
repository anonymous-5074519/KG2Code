import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # force GPU id=1 for both vLLM and torch

import json
from tqdm import tqdm
import torch
import sys
import re
import pickle
from pathlib import Path

from vllm import LLM, SamplingParams


# -----------------------------
# Load cached resources
# -----------------------------
endict = pickle.load(open('endict.pkl', 'rb'))
out_en_re = pickle.load(open("out_en_re.pkl", "rb"))
in_en_re = pickle.load(open("in_en_re.pkl", "rb"))


# -----------------------------
# Configs
# -----------------------------
# wikidata5m_transductive, wikidata5m_inductive
DATA = 'wikidata5m_transductive'
# llm: Meta-Llama-3.1-8B-Instruct, deepseek-coder-7b-instruct-v1.5, CodeLlama-7b-Instruct-hf, DeepSeek-Coder-V2-Lite-Instruct,Qwen2.5-Coder-7B-Instruct
LLM_NAME = 'DeepSeek-Coder-V2-Lite-Instruct'
BATCH_SIZE = 64  # <-- explicit batch size knob

# I/O paths
test = json.load(open(f'../graph/{DATA}/test.json', 'r', encoding='utf-8'))
result_path = f'{DATA}/{LLM_NAME}/infer-code.json'
os.makedirs(f'{DATA}/{LLM_NAME}', exist_ok=True)

# Local model paths
LLM_PATH = f'../../checkpoint-code/{LLM_NAME}/merge'  # base model path


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
# vLLM Model Loader
# -----------------------------
def build_engine():
    """
    Build the vLLM engine and (optionally) enable a LoRA adapter.
    We rely on llm.get_tokenizer() to retrieve EOS/Pad token ids if needed.
    """
    llm = LLM(
        model=LLM_PATH,
        dtype="half",                 # fp16 for speed/memory
        tensor_parallel_size=1,       # single visible GPU (id=1)
        trust_remote_code=True,       # allow custom model code if needed
        gpu_memory_utilization=0.9,
        #max_model_len=32768
        max_model_len=4096
    )
    return llm


# -----------------------------
# Batch Inference via vLLM
# -----------------------------
def llm_response_batch(prompts, llm, sampling_params):
    """
    Generate model responses for a batch of prompts using vLLM.
    Attach LoRA via LoRARequest only when PEFT_PATH is provided.
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
# Main
# -----------------------------
def main():
    # Build vLLM engine and tokenizer
    llm = build_engine()
    tokenizer = llm.get_tokenizer()

    # vLLM sampling configuration
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
        batch_inputs = [s["input"] for s in batch_samples]

        # Inference with vLLM
        batch_responses = llm_response_batch(batch_inputs, llm, sampling_params)

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
            # not save because some of the samples have too many answer entities
            #sample["answer_mid"] = ans_e
            #sample["answer_name"] = ans_n

            # Save raw response
            sample['response'] = response

            # Extract predicted answers from lines like: answer.append("...")
            re_answer = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('answer.append'):
                    m = re.findall(r'answer\.append\("(.+)"\)', line)
                    if m:
                        cand = m[0].strip().lower()
                        if cand and cand not in re_answer:
                            re_answer.append(cand)
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
    print('*' * 30, 'Final Metrics', '*' * 30)
    print('*' * 30, 'Final Raw Metrics', '*' * 30)
    print('Hits@1: {}'.format(hit1_r / len(test)))
    print('Hits@3: {}'.format(hit3_r / len(test)))
    print('Hits@10: {}'.format(hit10_r / len(test)))
    print('*' * 30, 'Final Filtered Metrics', '*' * 30)
    print('Hits@1: {}'.format(hit1_f / len(test)))
    print('Hits@3: {}'.format(hit3_f / len(test)))
    print('Hits@10: {}'.format(hit10_f / len(test)))

    # Persist results
    json.dump(data, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
