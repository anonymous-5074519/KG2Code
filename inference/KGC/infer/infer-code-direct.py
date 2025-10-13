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
DATA = 'wikidata5m_inductive'
# llm: Meta-Llama-3.1-8B-Instruct, DeepSeek-Coder-V2-Lite-Instruct,CodeLlama-7b-Instruct-hf 
LLM_NAME = 'DeepSeek-Coder-V2-Lite-Instruct'
BATCH_SIZE = 64  # <-- explicit batch size knob

# I/O paths
test = json.load(open(f'../graph/{DATA}/test.json', 'r', encoding='utf-8'))
result_path = f'{DATA}/{LLM_NAME}/infer-code-direct.json'
os.makedirs(f'{DATA}/{LLM_NAME}', exist_ok=True)

# Local model / adapter paths
LLM_PATH = f'../../../../pretrain/{LLM_NAME}'

demo_prompt='''import networkx as nx
graph = nx.Graph()
graph.add_node("Whittier Mill Village")
graph.add_node("historic district in the United States")
graph.add_edge("Whittier Mill Village", "historic district in the United States", relation="instance of")
def KGC(triple, graph):
    """
    complete the missing part of the triple based on the knowledge graph
    """
    triple = (Whittier Mill Village, country, [MASK])
    answer = []
    """
    First, the graph indicates that Whittier Mill Village is classified as an "historic district in the United States." This classification directly implies that it is located within the United States. Since the masked triple is asking for the country in which Whittier Mill Village is situated, and the graph already specifies that it is a historic district in the United States, we can confidently conclude that the answer is the United States. Therefore, the answer is United States.
    """
    answer.append("United States")
    return answer
    
import networkx as nx
graph = nx.Graph()
graph.add_node("Social Democratic Party")
graph.add_node("Portugal")
graph.add_node("Prime Minister of Portugal")
graph.add_node("human")
graph.add_node("Catholicism")
graph.add_node("University of Lisbon")
graph.add_node("Lisbon")
graph.add_node("Member of the European Parliament")
graph.add_node("Strasbourg")
graph.add_node("City of Brussels")
graph.add_node("Pedro")
graph.add_node("Santana")
graph.add_node("Pedro Santana Lopes")
graph.add_node("Lopes")
graph.add_node("Portuguese")
graph.add_edge("Pedro Santana Lopes", "City of Brussels", relation="work location")
graph.add_edge("Pedro Santana Lopes", "Prime Minister of Portugal", relation="position held")
graph.add_edge("Pedro Santana Lopes", "Portuguese", relation="languages spoken, written or signed")
graph.add_edge("Pedro Santana Lopes", "human", relation="instance of")
graph.add_edge("Pedro Santana Lopes", "Lopes", relation="second family name in Spanish name")
graph.add_edge("Pedro Santana Lopes", "Portugal", relation="country of citizenship")
graph.add_edge("Pedro Santana Lopes", "Catholicism", relation="religion or worldview")
graph.add_edge("Pedro Santana Lopes", "University of Lisbon", relation="educated at")
graph.add_edge("Pedro Santana Lopes", "Lisbon", relation="place of birth")
graph.add_edge("Pedro Santana Lopes", "Member of the European Parliament", relation="position held")
graph.add_edge("Pedro Santana Lopes", "Santana", relation="family name")
graph.add_edge("Pedro Santana Lopes", "Strasbourg", relation="work location")
graph.add_edge("Pedro Santana Lopes", "Pedro", relation="given name")
graph.add_edge("Pedro Santana Lopes", "Social Democratic Party", relation="member of political party")
def KGC(triple, graph):
    """
    complete the missing part of the triple based on the knowledge graph
    """
    triple = (Pedro Santana Lopes, occupation, [MASK])
    answer = []
    """
    To determine the occupation of Pedro Santana Lopes, we can analyze the information provided in the graph and apply our pre-existing knowledge. 1. **Graph Analysis**: The graph indicates that Pedro Santana Lopes has held significant political positions, specifically as the Prime Minister of Portugal and as a Member of the European Parliament. These roles are inherently political and suggest a strong involvement in governance and public service. 2. **Political Affiliation**: The graph also states that he is a member of the Social Democratic Party, which further emphasizes his political career. This affiliation typically aligns with individuals who are active in politics and governance. 3. **Legal Background**: While the graph does not explicitly mention a legal profession, it is common for politicians, especially those in high-ranking positions like Prime Minister, to have a background in law. Many politicians are trained as lawyers or jurists, which is often a prerequisite for understanding legislative processes and governance. 4. **Conclusion**: Given the political roles he has held and the common practice of politicians having legal training, it is reasonable to conclude that Pedro Santana Lopes's occupation includes being a lawyer, jurist, and politician. Therefore, the answer is lawyer, jurist, and politician.
    """
    answer.append("lawyer")
    answer.append("jurist")
    answer.append("politician")
    return answer
    
import networkx as nx
graph = nx.Graph()
graph.add_node("film")
graph.add_node("India")
graph.add_node("Hindi")
graph.add_node("Hisaab Khoon Ka")
graph.add_edge("Hisaab Khoon Ka", "India", relation="country of origin")
graph.add_edge("Hisaab Khoon Ka", "Hindi", relation="original language of film or TV show")
graph.add_edge("Hisaab Khoon Ka", "film", relation="instance of")
def KGC(triple, graph):
    """
    complete the missing part of the triple based on the knowledge graph
    """
    triple = (Hisaab Khoon Ka, cast member, [MASK])
    answer = []
    """
    To determine the cast members of "Hisaab Khoon Ka," we start by analyzing the information provided in the graph. The graph indicates that "Hisaab Khoon Ka" is a film from India and is in Hindi. However, it does not provide specific details about the cast members. Given my pre-existing knowledge, I can recall that "Hisaab Khoon Ka" is a Hindi film that features several well-known actors. Notably, Saeed Jaffrey, Raj Babbar, Mithun Chakraborty, Satish Shah, Amrish Puri, Mandakini, and Poonam Dhillon are prominent figures in the Indian film industry and have appeared in various films during the era when "Hisaab Khoon Ka" was released. Since the graph does not specify any cast members, I rely on my knowledge of the film and its context to identify these actors as likely cast members. Therefore, the answer includes Saeed Jaffrey, Raj Babbar, Mithun Chakraborty, Satish Shah, Amrish Puri, Mandakini, and Poonam Dhillon, as they are all associated with the film and fit the criteria of being cast members.
    """
    answer.append("Saeed Jaffrey")
    answer.append("Raj Babbar")
    answer.append("Mithun Chakraborty")
    answer.append("Satish Shah")
    answer.append("Amrish Puri")
    answer.append("Mandakini")
    answer.append("Poonam Dhillon")
    return answer
    
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
        max_model_len=8192
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
        batch_inputs = [demo_prompt+s["input"] for s in batch_samples]
        print(batch_inputs[0])

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
