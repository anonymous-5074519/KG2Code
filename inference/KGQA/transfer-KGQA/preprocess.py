import json
import re
import os

data = json.load(open('dataset/grailqa/grailqa.json','r',encoding='utf-8'))

with open('prompt/kgqa_input.txt','r',encoding='utf-8') as f:
    qa_input_prompt = f.read()
with open('prompt/kgqa_output.txt','r',encoding='utf-8') as f:
    qa_output_prompt = f.read()
with open('prompt/base.txt','r',encoding='utf-8') as f:
    base_prompt = f.read()

process_data = []
for sample in data:
    triplestr = sample["triples"]
    triplelist = []
    matches = re.findall(r'\((.*?)\)', triplestr)
    for m in matches:
        parts = [p.strip() for p in m.split(',')]
        if len(parts) != 3:
            continue
        triplelist.append(parts)
    sample["graph_extend_name"] = triplelist

    seen = set()
    entities = []
    for h, r, t in triplelist:
        for e in (h, t):
            if e not in seen:
                seen.add(e)
                entities.append(e)

    prompt = base_prompt + '\n'
    for e in entities:
        prompt += f'graph.add_node("{e}")\n'

    for h, r, t in triplelist:
        prompt += f'graph.add_edge("{h}", "{t}", relation="{r}")\n'

    # kgqa
    sample["input"] = prompt + qa_input_prompt.format(question=sample["question"])
    sample["answer_name"] = sample["answer"]
    process_data.append(sample)

os.makedirs('graph/grailqa',exist_ok=True)
json.dump(process_data, open('graph/grailqa/test.json', 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
