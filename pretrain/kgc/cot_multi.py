import time
import json
import re
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from openai import OpenAI

random.seed(42)
# set client
client=OpenAI(api_key='sk-DJrylVD3AU0H2PVcdTD88vj9paUfi0kYCkWXBJF85e4Ut6lt',base_url = "https://35.aigcbest.top/v1")

cot_prompt='''Your task is to generate the Chain-of-Thought (CoT) reasoning process for a link prediction task, given the masked triple, the graph, and the answer. The reasoning should integrate both the structural information from the graph and your pre-existing knowledge. If the graph information is incomplete or irrelevant, actively supplement it with your own knowledge to arrive at the answer.
Masked Triple: ([MASK], spouse, Charles the Simple)
Graph: (Charles the Simple, father, Louis the Stammerer) (Charles the Simple, given name, Karel) (Charles the Simple, child, Gisela of France) (Louis IV of France, faple) (Charles the Simple, sibling, Ermentrude of France) (Carloman II of France, sibling, Charles the Simple) (Louis III of France, sibling, Charles the Simple, sibling, Louis III of France) (Charles the Simple, spouse, Frederuna) (Charles the Simple, family, Carolingian dynasty) (Charles the Simple, place of death,  Simple, occupation, monarch) (Charles the Simple, instance of, human) (Charles the Simple, sibling, Carloman II of France) (Charles the Simple, described by st Encyclopedia) (Adelaide of Paris, child, Charles the Simple) (Ermentrude of France, sibling, Charles the Simple) (Charles the Simple, spouse, Eadgifu of Wessrer, child, Charles the Simple) (Charles the Simple, mother, Adelaide of Paris) (Charles the Simple, child, Louis IV of France) (Charles the Simple, country of (Gisela of France, father, Charles the Simple)
Answer: Eadgifu of Wessex|Frederuna
CoT: First, the graph explicitly states Charles the Simple has two spouses: Frederuna and Eadgifu of Wessex. Therefore, the answer is Eadgifu of Wessex and Frederuna.

Your task is to generate the Chain-of-Thought (CoT) reasoning process for a link prediction task, given the masked triple, the graph, and the answer. The reasoning should integrate both the structural information from the graph and your pre-existing knowledge. If the graph information is incomplete or irrelevant, actively supplement it with your own knowledge to arrive at the answer.
Masked Triple: (X-Alfonso, occupation, [MASK])
Graph: (X-Alfonso, country of citizenship, Cuba) (X-Alfonso, genre, jazz fusion) (Habana Blues, composer, X-Alfonso) (X-Alfonso, place of birth, Havana) (X-Alfonso, instance of, human)
Answer: musician|singer
CoT: First, the graph indicates that X-Alfonso is associated with the genre of jazz fusion, which is a style of music. Second, X-Alfonso is noted as the composer of "Habana Blues," which further suggests involvement in music. These details strongly imply that X-Alfonso's occupation is related to music. Given this context, the most likely occupations for X-Alfonso are musician and singer, as these roles align with composing music and being associated with a musical genre. Therefore, the answer is musician and singer.

Your task is to generate the Chain-of-Thought (CoT) reasoning process for a link prediction task, given the masked triple, the graph, and the answer. The reasoning should integrate both the structural information from the graph and your pre-existing knowledge. If the graph information is incomplete or irrelevant, actively supplement it with your own knowledge to arrive at the answer.
Masked Triple: (Karwacja, located in time zone, [MASK])
Graph: (Karwacja, located in the administrative territorial entity, Gmina Sierakowice) (Karwacja, country, Poland)
Answer: UTC+01:00|UTC+02:00
CoT: First, the graph provides information that Karwacja is located in Poland. To determine the time zone, we need to consider Poland's standard time zones. Poland typically operates under Central European Time (CET), which is UTC+01:00, during the standard time period. Additionally, Poland observes daylight saving time, known as Central European Summer Time (CEST), which is UTC+02:00. Therefore, depending on the time of year, Karwacja would be located in either UTC+01:00 or UTC+02:00. Thus, the answer is UTC+01:00 and UTC+02:00.

Your task is to generate the Chain-of-Thought (CoT) reasoning process for a link prediction task, given the masked triple, the graph, and the answer. The reasoning should integrate both the structural information from the graph and your pre-existing knowledge. If the graph information is incomplete or irrelevant, actively supplement it with your own knowledge to arrive at the answer.
Masked Triple: {mask}
Graph: {graph}
Answer: {answer}
CoT: '''

data = json.load(open('kgc-data.json', 'r', encoding='utf-8'))
random.shuffle(data)

num1 = num2 = 0
data1 = []
maskset=set()

for sample in data:
    # avoid redundant masked triples or triples without subgraph
    if tuple(sample["masked triple name"]) in maskset or len(sample["subgraph"])==0:
        continue
    maskset.add(tuple(sample["masked triple name"]))
    if sample["masked triple name"][0] == "[MASK]":
        if num1 < 100000:
            data1.append(sample)
            num1 += 1
    else:
        if num2 < 100000:
            data1.append(sample)
            num2 += 1
    if num1 >= 100000 and num2 >= 100000:
        break

def process_sample(sample):
    mask = '(' + ', '.join(sample["masked triple name"]) + ')'
    graph_str = ' '.join(f'({g[0]}, {g[1]}, {g[2]})' for g in sample["subgraph name"]).strip()
    answer = '|'.join(sample["answer name"][:10])
    prompt = cot_prompt.format(mask=mask, graph=graph_str, answer=answer)

    retries = 0
    while retries < 10:
        try:
            res = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
            )
            cot = res.choices[0].message.content
            cot = re.sub(r'\n+', ' ', cot)
            cot = re.sub(r'\s+', ' ', cot)
            sample["CoT"] = cot
            return sample
        except Exception as e:
            print(f"[{sample['masked triple name']}] Error: {e}")
            retries += 1
            time.sleep(60)
    return None

if __name__ == "__main__":
    max_processes = min(20, cpu_count())
    batch_size = 10000
    data2 = []

    with Pool(processes=max_processes) as pool:
        for i, result in enumerate(tqdm(pool.imap(process_sample, data1), total=len(data1))):
            if result and result.get("CoT"):
                data2.append(result)
                
            if len(data2) > 0 and len(data2) % batch_size == 0:
                filename = f"kgc_cot_{len(data2)}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data2, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(data2)} samples to {filename}")

    if data2:
        filename = f"kgc_cot.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data2, f, indent=2, ensure_ascii=False)
        print(f"Saved final {len(data2)} samples to {filename}")
