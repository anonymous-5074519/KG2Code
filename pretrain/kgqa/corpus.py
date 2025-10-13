import json

data=json.load(open('all_question_graph_extend.json','r',encoding='utf-8'))

with open("prompt/base.txt","r",encoding='utf-8') as f:
    base_prompt=f.read()
    
with open("prompt/kgqa_input.txt","r",encoding='utf-8') as f:
    kgqa_input=f.read()
    
with open("prompt/kgqa_output.txt","r",encoding='utf-8') as f:
    kgqa_output=f.read()

corpus=[]
for sample in data:
    enset=set()
    graphstr=""
    for t in sample["extend graph name"]:
        enset.add(t[0])
        enset.add(t[2])
        graphstr=graphstr+'('+', '.join(t)+') '
    graphstr=graphstr.strip()
    instruction=base_prompt+'\n'
    for e in enset:
        instruction=instruction+'graph.add_node("{entity}")\n'.format(entity=e)
    for t in sample["extend graph name"]:
        instruction=instruction+'graph.add_edge("{e1}", "{e2}", relation="{r}")\n'.format(e1=t[0],e2=t[2],r=t[1])
    instruction=instruction+kgqa_input.format(question=sample["question"])
    output=kgqa_output.format(cot=sample["cot"])+'\n'
    for a in sample["answer name"]:
        output=output+'    answer.append("{answer}")\n'.format(answer=a)
    output=output+"    return answer"
    sample_dict=dict()
    sample_dict["instruction"]=instruction
    sample_dict["input"]=""
    sample_dict["output"]=output
    corpus.append(sample_dict)

json.dump(corpus,open('kgqa_corpus.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
    
    
    
    
