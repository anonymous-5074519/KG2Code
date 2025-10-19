from openai import OpenAI
import time
import json
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

data = json.load(open('all_question.json', 'r', encoding='utf-8'))

def init_client():
    return OpenAI(
        api_key='YOUR API KEY'
    )

cot_prompt_template='''Your task is to generate the Chain-of-Thought(CoT) reasoning process given the question, the question-related graph, and the answer.
Question: Is it true that H. Justin Davidson was born in Gentryville, is a citizen of Tanganyika, and was educated at Carnegie Mellon University?
Graph: (H. Justin Davidson, place of birth, Gentryville) (H. Justin Davidson, country of citizenship, United States) (H. Justin Davidson, educated at, Carnegie Mellon University)
Answer: no
CoT: First, H. Justin Davidson was born in Gentryville, which confirms the first part of the statement. Second, H. Justin Davidson is a citizen of the United States, not Tanganyika, which contradicts the second part of the statement. Third, H. Justin Davidson was educated at Carnegie Mellon University, which confirms the third part of the statement. Therefore, since the second part of the statement is false, the answer is no.

Your task is to generate the Chain-of-Thought(CoT) reasoning process given the question, the question-related graph, and the answer.
Question: How many stations adjacent to the Pennsauken Transit Center are owned by New Jersey Transit?
Graph: (Pennsauken Route 73, adjacent station, Pennsauken Transit Center) (Pennsauken Route 73, owned by, New Jersey Transit) (Cherry Hill station, adjacent station, Pennsauken Transit Center) (Cherry Hill station, owned by, New Jersey Transit) (36th Street, adjacent station, Pennsauken Transit Center) (36th Street, owned by, New Jersey Transit)
Answer: 3
CoT: First, Pennsauken Route 73, Cherry Hill station, and 36th Street are all adjacent to the Pennsauken Transit Center. Second, Pennsauken Route 73, Cherry Hill station, and 36th Street are all owned by New Jersey Transit. Therefore, there are three stations adjacent to the Pennsauken Transit Center that are owned by New Jersey Transit. Therefore, the answer is 3.

Your task is to generate the Chain-of-Thought(CoT) reasoning process given the question, the question-related graph, and the answer.
Question: What organizations or groups is the educational institution, where Mike Brubaker studied, a member of?
Graph: (West Virginia University, member of, Scholarly Publishing and Academic Resources Coalition) (West Virginia University, member of, Center for Research Libraries) (West Virginia University, member of, Digital Library Federation) (Mike Brubaker, educated at, West Virginia University)
Answer: Digital Library Federation|Scholarly Publishing and Academic Resources Coalition|Center for Research Librarie
CoT: First, Mike Brubaker was educated at West Virginia University. Second, West Virginia University is a member of several organizations or groups, including the Scholarly Publishing and Academic Resources Coalition, the Center for Research Libraries, and the Digital Library Federation. Therefore, the answer is Digital Library Federation, Scholarly Publishing and Academic Resources Coalition, and Center for Research Libraries.

Your task is to generate the Chain-of-Thought(CoT) reasoning process given the question, the question-related graph, and the answer.
Question: {question}
Graph: {graph}
Answer: {answer}
CoT: '''

def process_sample(sample):
    client = init_client()

    ques = sample["question"]
    graph_str = ' '.join(f'({g[0]}, {g[1]}, {g[2]})' for g in sample["subgraph name"]).strip()
    ansstr = '|'.join(sample["answer name"])

    prompt = cot_prompt_template.format(question=ques, graph=graph_str, answer=ansstr)
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
            sample["cot"] = cot
            return sample
        except Exception as e:
            print(f"Error on sample '{sample.get('question', '')[:50]}...': {e}")
            retries += 1
            time.sleep(60)
    sample["cot"] = ''
    return sample

if __name__ == "__main__":
    data_to_process = data
    num_workers = min(20, cpu_count())
    checkpoint_interval = 10000

    results = []
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(data_to_process), desc="Processing") as pbar:
            for i, result in enumerate(pool.imap_unordered(process_sample, data_to_process), 1):
                results.append(result)
                pbar.update()
                if i % checkpoint_interval == 0:
                    filename = f"all_question_cot_{i}.json"
                    json.dump(results, open(filename, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
                    print(f"Checkpoint saved to {filename} after {i} samples.\n")

    json.dump(results, open("all_question_cot.json", 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

