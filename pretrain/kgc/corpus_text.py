import json

data=json.load(open('kgc_cot.json','r',encoding='utf-8'))

input_prompt='''Please infer the missing part of the triple based on the subgraph retrieved from the knowledge graph. First, provide your Chain-of-Thought (CoT) reasoning process. At the end, list all plausible answers in a single line, starting with "[MASK]: " and separating each answer by "|" as follows: [MASK]: First answer|Second answer|Third answer ...
Subgraph: {knowledge}
Triple: {triple}
'''

output_prompt='''CoT: {cot}
[MASK]: {answer}'''

corpus=[]
for sample in data:
    graphstr=""
    for t in sample["subgraph name"]:
        graphstr=graphstr+'('+', '.join(t)+') '
    graphstr=graphstr.strip()
    instruction=input_prompt.format(knowledge=graphstr,triple='('+', '.join(sample["masked triple name"])+')')
    output=output_prompt.format(cot=sample["CoT"],answer='|'.join(sample["answer name"][:10]))
    sample_dict=dict()
    sample_dict["instruction"]=instruction
    sample_dict["input"]=""
    sample_dict["output"]=output
    corpus.append(sample_dict)

json.dump(corpus,open('kgc_text_corpus.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)