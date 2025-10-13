import json

data=json.load(open('all_question_graph_extend.json','r',encoding='utf-8'))

input_prompt='''Please answer the question based on the subgraph retrieved from the knowledge graph. First, provide your Chain-of-Thought (CoT) reasoning process. At the end, list all answers in a single line, starting with "Answer: " and separating each answer by "|" as follows: Answer: First answer|Second answer|Third answer ...
Subgraph: {knowledge}
Question: {ques}
'''

output_prompt='''CoT: {cot}
Answer: {answer}'''

corpus=[]
for sample in data:
    enset=set()
    graphstr=""
    for t in sample["extend graph name"]:
        enset.add(t[0])
        enset.add(t[2])
        graphstr=graphstr+'('+', '.join(t)+') '
    graphstr=graphstr.strip()
    instruction=input_prompt.format(knowledge=graphstr,ques=sample["question"])
    output=output_prompt.format(cot=sample["cot"],answer='|'.join(sample["answer name"]))
    sample_dict=dict()
    sample_dict["instruction"]=instruction
    sample_dict["input"]=""
    sample_dict["output"]=output
    corpus.append(sample_dict)

json.dump(corpus,open('kgqa_text_corpus.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)