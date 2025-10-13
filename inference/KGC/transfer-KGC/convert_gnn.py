import json
import os

data = json.load(open('graph/WN18RR/test.json','r',encoding='utf-8'))

index=0
process=[]
for sample in data:
    temp=dict()
    temp["id"]=str(index)
    temp["question"]="What is the [MASK] entity in the following triple: {triple}?".format(triple=sample["mask_triple"])
    temp["answer"]=[sample["gt_name"]]
    if sample["type"]=='head':
        temp["q_entity"]=[sample["triple_name"].split('|')[2]]
    if sample["type"]=='tail':
        temp["q_entity"]=[sample["triple_name"].split('|')[0]]
    temp["a_entity"]=[sample["gt_name"]]
    temp["graph"]=sample["graph_name"]
    temp["choices"]=[]
    process.append(temp)
    index+=1
    
os.makedirs('GNN/WN18RR',exist_ok=True)
with open('GNN/WN18RR/test.jsonl', "w", encoding="utf-8") as f:
    for item in process:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")    