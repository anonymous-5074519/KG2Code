import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # force GPU id=2 for vLLM

import json
import re
import sys
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams

# -----------------------------
# Configs
# -----------------------------
# QALD-9, QALD-10, MintQA-Pop
DATA = 'QALD-10'
# Meta-Llama-3.1-8B-Instruct, deepseek-coder-7b-instruct-v1.5, CodeLlama-7b-Instruct-hf,DeepSeek-Coder-V2-Lite-Instruct
LLM_NAME = 'DeepSeek-Coder-V2-Lite-Instruct'
BATCH_SIZE = 64  # <-- explicit batch size knob
# I/O paths
test = json.load(open(f'../graph/{DATA}/test.json', 'r', encoding='utf-8'))
result_path = f'{DATA}/{LLM_NAME}/answer-code-direct.json'
os.makedirs(f'{DATA}/{LLM_NAME}', exist_ok=True)

# Local model / adapter paths
LLM_PATH = f'../../../../pretrain/{LLM_NAME}'         # base model path

demo_prompt='''import networkx as nx
graph = nx.Graph()
graph.add_node("Adams Memorial Building")
graph.add_node("United States")
graph.add_node("Lily Lake")
graph.add_node("Jacob Lindley")
graph.add_node("Planet B-Boy")
graph.add_node("business")
graph.add_node("WNJB-FM")
graph.add_node("George Junior Republic")
graph.add_node("Mackinac Transportation Company")
graph.add_node("Pastura")
graph.add_node("English")
graph.add_node("Whitey Moore")
graph.add_node("Nicholas Raymond Cerio")
graph.add_node("Jean Klock Park")
graph.add_node("John Whitaker")
graph.add_node("Bobby Orrock")
graph.add_node("Rocky Mountains")
graph.add_node("Brian Dietzen")
graph.add_node("Waste No Food")
graph.add_node("Sonny's BBQ")
graph.add_node("Thomas Morrow Reavley")
graph.add_node("Normie Glick")
graph.add_node("Daniel Shulman")
graph.add_node("Community Foundation for Northeast Georgia")
graph.add_node("beer in San Diego County, California")
graph.add_node("Palestine")
graph.add_node("Nampa High School")
graph.add_node("Ben Hill County Courthouse")
graph.add_node("Winter Park")
graph.add_node("Lanesville Township")
graph.add_edge("Pastura", "United States", relation="country")
graph.add_edge("Jacob Lindley", "United States", relation="country of citizenship")
graph.add_edge("Rocky Mountains", "United States", relation="country")
graph.add_edge("George Junior Republic", "United States", relation="country")
graph.add_edge("Sonny's BBQ", "United States", relation="country")
graph.add_edge("Bobby Orrock", "United States", relation="country of citizenship")
graph.add_edge("Sonny's BBQ", "business", relation="instance of")
graph.add_edge("Waste No Food", "United States", relation="country")
graph.add_edge("Whitey Moore", "United States", relation="country of citizenship")
graph.add_edge("Thomas Morrow Reavley", "United States", relation="country of citizenship")
graph.add_edge("beer in San Diego County, California", "United States", relation="country")
graph.add_edge("WNJB-FM", "United States", relation="country")
graph.add_edge("United States", "United States", relation="country")
graph.add_edge("Community Foundation for Northeast Georgia", "United States", relation="country")
graph.add_edge("Lanesville Township", "United States", relation="country")
graph.add_edge("Planet B-Boy", "United States", relation="country of origin")
graph.add_edge("Brian Dietzen", "United States", relation="country of citizenship")
graph.add_edge("Palestine", "United States", relation="country")
graph.add_edge("Jean Klock Park", "United States", relation="country")
graph.add_edge("Nampa High School", "United States", relation="country")
graph.add_edge("Nicholas Raymond Cerio", "United States", relation="country of citizenship")
graph.add_edge("Daniel Shulman", "United States", relation="country of citizenship")
graph.add_edge("John Whitaker", "United States", relation="country of citizenship")
graph.add_edge("United States", "English", relation="official language")
graph.add_edge("Adams Memorial Building", "United States", relation="country")
graph.add_edge("Mackinac Transportation Company", "United States", relation="country")
graph.add_edge("Ben Hill County Courthouse", "United States", relation="country")
graph.add_edge("Lily Lake", "United States", relation="country")
graph.add_edge("Normie Glick", "United States", relation="country of citizenship")
graph.add_edge("Sonny's BBQ", "Winter Park", relation="headquarters location")
def KGQA(question, graph):
    """
    answer the question based on the knowledge graph
    """
    question = "What is the official language of the country where Sonny's BBQ is located?"
    answer = []
    """
    First, Sonny's BBQ is located in the United States, as indicated in the graph. Second, the graph states that the official language of the United States is English. Therefore, since Sonny's BBQ is in the United States, the official language of the country where Sonny's BBQ is located is English. Thus, the answer is English.
    """
    answer.append("English")
    return answer

import networkx as nx
graph = nx.Graph()
graph.add_node("United States")
graph.add_node("Elburn")
graph.add_node("George Washington")
graph.add_node("Bill Sweatt")
graph.add_node("Martin Luther King Jr.")
graph.add_node("Independence Day")
graph.add_node("human")
graph.add_node("Labor Day")
graph.add_node("New Year's Day")
graph.add_node("Christopher Columbus")
graph.add_node("ice hockey")
graph.add_node("Indigenous People's Day")
graph.add_node("Bill")
graph.add_node("Washington's Birthday")
graph.add_node("federal holiday in the United States")
graph.add_node("Vancouver Canucks")
graph.add_node("Spanish national day")
graph.add_node("Thanksgiving")
graph.add_node("Veterans Day")
graph.add_node("Columbus Day")
graph.add_node("Memorial Day")
graph.add_node("Chicago Blackhawks")
graph.add_node("voyages by Christopher Columbus")
graph.add_node("winger")
graph.add_node("Martin Luther King Jr. Day")
graph.add_node("public holidays in the United States")
graph.add_edge("Martin Luther King Jr. Day", "federal holiday in the United States", relation="instance of")
graph.add_edge("Bill Sweatt", "winger", relation="position played on team / speciality")
graph.add_edge("Washington's Birthday", "United States", relation="country")
graph.add_edge("Labor Day", "public holidays in the United States", relation="instance of")
graph.add_edge("United States", "Veterans Day", relation="public holiday")
graph.add_edge("United States", "New Year's Day", relation="public holiday")
graph.add_edge("Columbus Day", "Christopher Columbus", relation="named after")
graph.add_edge("United States", "Columbus Day", relation="public holiday")
graph.add_edge("Chicago Blackhawks", "United States", relation="country")
graph.add_edge("Bill Sweatt", "ice hockey", relation="sport")
graph.add_edge("Indigenous People's Day", "Columbus Day", relation="different from")
graph.add_edge("Bill Sweatt", "Chicago Blackhawks", relation="drafted by")
graph.add_edge("United States", "Labor Day", relation="public holiday")
graph.add_edge("Washington's Birthday", "George Washington", relation="named after")
graph.add_edge("Washington's Birthday", "public holidays in the United States", relation="instance of")
graph.add_edge("Bill Sweatt", "Vancouver Canucks", relation="member of sports team")
graph.add_edge("Martin Luther King Jr. Day", "Martin Luther King Jr.", relation="named after")
graph.add_edge("Spanish national day", "Columbus Day", relation="subclass of")
graph.add_edge("United States", "Martin Luther King Jr. Day", relation="public holiday")
graph.add_edge("Bill Sweatt", "human", relation="instance of")
graph.add_edge("United States", "Independence Day", relation="public holiday")
graph.add_edge("Labor Day", "United States", relation="country")
graph.add_edge("United States", "Memorial Day", relation="public holiday")
graph.add_edge("Bill Sweatt", "United States", relation="country of citizenship")
graph.add_edge("United States", "Thanksgiving", relation="public holiday")
graph.add_edge("United States", "Washington's Birthday", relation="public holiday")
graph.add_edge("Bill Sweatt", "Elburn", relation="place of birth")
graph.add_edge("Columbus Day", "voyages by Christopher Columbus", relation="commemorates")
graph.add_edge("Bill Sweatt", "Bill", relation="given name")
graph.add_edge("Columbus Day", "federal holiday in the United States", relation="instance of")
def KGQA(question, graph):
    """
    answer the question based on the knowledge graph
    """
    question = "Which public holidays are observed in the country where Bill Sweatt was drafted?"
    answer = []
    """
    First, I need to determine the country where Bill Sweatt was drafted. The graph indicates that he was drafted by the Chicago Blackhawks, which is associated with the United States. Next, I will identify the public holidays observed in the United States as listed in the graph. The holidays mentioned are Labor Day, New Year's Day, Memorial Day, Washington's Birthday, Independence Day, Veterans Day, Martin Luther King Jr. Day, Thanksgiving, and Columbus Day. Since all these holidays are observed in the United States, I can conclude that they are the public holidays observed in the country where Bill Sweatt was drafted. Therefore, the answer is Columbus Day, Veterans Day, Washington's Birthday, Labor Day, Independence Day, New Year's Day, Memorial Day, Martin Luther King Jr. Day, and Thanksgiving.
    """
    answer.append("Columbus Day")
    answer.append("Veterans Day")
    answer.append("Washington's Birthday")
    answer.append("Labor Day")
    answer.append("Independence Day")
    answer.append("New Year's Day")
    answer.append("Memorial Day")
    answer.append("Martin Luther King Jr. Day")
    answer.append("Thanksgiving")
    return answer

import networkx as nx
graph = nx.Graph()
graph.add_node("Malta")
graph.add_node("Gozo Aqueduct")
graph.add_node("1883 Maltese general election")
graph.add_node("Dora Rappard")
graph.add_node("Strada Stretta")
graph.add_node("list of hospitals in Malta")
graph.add_node("sports governing body")
graph.add_node("Piran")
graph.add_node("Vilnius")
graph.add_node("Donat Spiteri")
graph.add_node("Emma Xuerreb")
graph.add_node("Giovanni Carmine Pellerano")
graph.add_node("Joseph De Piro")
graph.add_node("Jacob Borg")
graph.add_node("Sacro Cuor Parish Church")
graph.add_node("Spinola Battery")
graph.add_node("1888 Maltese general election")
graph.add_node("Auberge d'Allemagne")
graph.add_node("Misra? G?ar il-Kbir")
graph.add_node("Eddie Fenech Adami")
graph.add_node("Valletta")
graph.add_node("Sinbad and the Eye of the Tiger")
graph.add_node("Dom Ambrose Agius")
graph.add_node("Anthony Bonanno")
graph.add_node("Qormi F.C.")
graph.add_node("Mediterranean Bank")
graph.add_node("Northern Region (Tramuntana)")
graph.add_node("Albert Garzia")
graph.add_node("Chucks Nwoko")
graph.add_node("Rhodes")
graph.add_node("Aquatic Sports Association of Malta")
graph.add_edge("Sacro Cuor Parish Church", "Malta", relation="country")
graph.add_edge("Giovanni Carmine Pellerano", "Malta", relation="country of citizenship")
graph.add_edge("Spinola Battery", "Malta", relation="country")
graph.add_edge("Dom Ambrose Agius", "Malta", relation="country of citizenship")
graph.add_edge("Albert Garzia", "Malta", relation="country of citizenship")
graph.add_edge("Sinbad and the Eye of the Tiger", "Malta", relation="filming location")
graph.add_edge("Eddie Fenech Adami", "Malta", relation="country of citizenship")
graph.add_edge("Aquatic Sports Association of Malta", "sports governing body", relation="instance of")
graph.add_edge("Joseph De Piro", "Malta", relation="country of citizenship")
graph.add_edge("Qormi F.C.", "Malta", relation="country")
graph.add_edge("Chucks Nwoko", "Malta", relation="country of citizenship")
graph.add_edge("Jacob Borg", "Malta", relation="country of citizenship")
graph.add_edge("Donat Spiteri", "Malta", relation="country of citizenship")
graph.add_edge("list of hospitals in Malta", "Malta", relation="country")
graph.add_edge("Aquatic Sports Association of Malta", "Malta", relation="country")
graph.add_edge("1888 Maltese general election", "Malta", relation="country")
graph.add_edge("Gozo Aqueduct", "Malta", relation="country")
graph.add_edge("Auberge d'Allemagne", "Malta", relation="country")
graph.add_edge("Valletta", "Piran", relation="twinned administrative body")
graph.add_edge("1883 Maltese general election", "Malta", relation="country")
graph.add_edge("Anthony Bonanno", "Malta", relation="country of citizenship")
graph.add_edge("Misra? G?ar il-Kbir", "Malta", relation="country")
graph.add_edge("Malta", "Valletta", relation="capital")
graph.add_edge("Strada Stretta", "Malta", relation="country of origin")
graph.add_edge("Valletta", "Vilnius", relation="twinned administrative body")
graph.add_edge("Mediterranean Bank", "Malta", relation="country")
graph.add_edge("Northern Region (Tramuntana)", "Malta", relation="country")
graph.add_edge("Emma Xuerreb", "Malta", relation="place of birth")
graph.add_edge("Dora Rappard", "Malta", relation="place of birth")
graph.add_edge("Valletta", "Rhodes", relation="twinned administrative body")
def KGQA(question, graph):
    """
    answer the question based on the knowledge graph
    """
    question = "What are the twinned administrative bodies of the capital city of the country where the Aquatic Sports Association of Malta is located?"
    answer = []
    """
    First, the Aquatic Sports Association of Malta is located in Malta, which has Valletta as its capital. Second, Valletta has several twinned administrative bodies, which are Piran, Vilnius, and Rhodes. Therefore, the twinned administrative bodies of Valletta, the capital city of Malta, are Vilnius, Rhodes, and Piran. Thus, the answer is Vilnius, Rhodes, and Piran.
    """
    answer.append("Vilnius")
    answer.append("Rhodes")
    answer.append("Piran")
    return answer
    
'''

# -----------------------------
# Metrics
# -----------------------------
def metrics_cal(answer, re_answer):
    """Compute Accuracy, Precision, Recall, F1, EM (case-insensitive, deduplicated)."""
    answer = list(set(a.lower() for a in answer))
    re_answer = list(set(a.lower() for a in re_answer))

    if not answer and not re_answer:
        return 1, 1, 1, 1, 1

    cor = 0
    acc_FLAG = False
    em_FLAG = True

    if answer:
        for a in answer:
            if a in re_answer:
                acc_FLAG = True
                cor += 1
            else:
                em_FLAG = False

        acc = 1 if acc_FLAG else 0
        precision = cor / len(re_answer) if re_answer else 0
        recall = cor / len(answer)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        em = 1 if em_FLAG and len(answer) == len(re_answer) else 0
    else:
        acc = precision = recall = f1 = em = 0

    return acc, precision, recall, f1, em

# -----------------------------
# vLLM Model Loader
# -----------------------------
def build_engine():
    """
    Build vLLM engine and (optionally) enable a LoRA adapter.
    We retrieve the tokenizer later via llm.get_tokenizer() for EOS id.
    """
    llm = LLM(
        model=LLM_PATH,
        dtype="half",                # fp16 for speed/memory
        tensor_parallel_size=1,      # single visible GPU (id=2)
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        #max_model_len=16384
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
        texts.append(r.outputs[0].text.strip() if r.outputs else "")
    return texts


# -----------------------------
# Main (Batch Inference)
# -----------------------------
def main():
    # Build engine and tokenizer
    llm = build_engine()
    tokenizer = llm.get_tokenizer()

    # vLLM sampling configuration (aligned with prior scripts)
    sampling_params = SamplingParams(
        temperature=0.01,
        top_k=40,
        top_p=0.9,
        n=1,
        max_tokens=1024,
        repetition_penalty=1.1,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    index = 0
    acc = precision = recall = f1 = EM = 0
    data = []

    pbar = tqdm(total=len(test))
    for start in range(0, len(test), BATCH_SIZE):
        batch_samples = test[start:start + BATCH_SIZE]
        batch_inputs = [demo_prompt+s["input"] for s in batch_samples]
        # Run vLLM generation
        batch_responses = llm_response_batch(batch_inputs, llm, sampling_params)

        # Parse responses and compute metrics
        for sample, response in zip(batch_samples, batch_responses):
            index += 1
            gold_answers = sample["answer_name"]

            # Extract predictions from lines like: answer.append("...")
            pred_answers = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('answer.append'):
                    m = re.findall(r'answer\.append\("(.+)"\)', line)
                    if m:
                        cand = m[0].strip()
                        if cand and cand not in pred_answers:
                            pred_answers.append(cand)

            # Metrics
            t_acc, t_prec, t_rec, t_f1, t_em = metrics_cal(gold_answers, pred_answers)
            acc += t_acc
            precision += t_prec
            recall += t_rec
            f1 += t_f1
            EM += t_em

            # Record per-sample data
            data.append({
                'question': sample['question'],
                'input': sample['input'],
                'answer': gold_answers,
                'graph': sample.get("graph_extend_name"),
                'response': response,
                'response_answer': pred_answers,
                'accuracy': t_acc,
                'precision': t_prec,
                'recall': t_rec,
                'f1': t_f1,
                'EM': t_em
            })

            # Streaming logs
            #print(batch_inputs)
            print(response)
            print('Current Accuracy: {}'.format(acc / index))
            print('Current Precision: {}'.format(precision / index))
            print('Current Recall: {}'.format(recall / index))
            print('Current F1: {}'.format(f1 / index))
            print('Current EM: {}'.format(EM / index))
            sys.stdout.flush()

        pbar.update(len(batch_samples))
    pbar.close()

    # Final metrics
    total = len(test)
    print('*' * 30, 'Final Metrics', '*' * 30)
    print('Accuracy: {}'.format(acc / total))
    print('Precision: {}'.format(precision / total))
    print('Recall: {}'.format(recall / total))
    print('F1: {}'.format(f1 / total))
    print('EM: {}'.format(EM / total))

    # Persist results
    json.dump(data, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
