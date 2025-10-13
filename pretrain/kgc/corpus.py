import json

data=json.load(open('kgc_cot.json','r',encoding='utf-8'))

with open("prompt/base.txt","r",encoding='utf-8') as f:
    base_prompt=f.read()
    
with open("prompt/kgc_input.txt","r",encoding='utf-8') as f:
    kgc_input=f.read()
    
with open("prompt/kgc_output.txt","r",encoding='utf-8') as f:
    kgc_output=f.read()
    
corpus=[]
for sample in data:
    enset=set()
    for t in sample["subgraph name"]:
        enset.add(t[0])
        enset.add(t[2])
    instruction=base_prompt+'\n'
    for e in enset:
        instruction=instruction+'graph.add_node("{entity}")\n'.format(entity=e)
    for t in sample["subgraph name"]:
        instruction=instruction+'graph.add_edge("{e1}", "{e2}", relation="{r}")\n'.format(e1=t[0],e2=t[2],r=t[1])
    instruction=instruction+kgc_input.format(triple='('+', '.join(sample["masked triple name"])+')')
    output=kgc_output.format(cot=sample["CoT"])+'\n'
    # select top 10 as the answers in training
    for a in sample["answer name"][:10]:
        output=output+'    answer.append("{answer}")\n'.format(answer=a)
    output=output+"    return answer"
    sample_dict=dict()
    sample_dict["instruction"]=instruction
    sample_dict["input"]=""
    sample_dict["output"]=output
    corpus.append(sample_dict)

json.dump(corpus,open('kgc_corpus.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)