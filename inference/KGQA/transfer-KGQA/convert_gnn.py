import json
import os

data = json.load(open('graph/grailqa/test.json','r',encoding='utf-8'))
origin = json.load(open('dataset/grailqa/grailqa_v1.0_dev.json','r',encoding='utf-8'))

index=0
process=[]
for sample in data:
    temp=dict()
    temp["id"]=str(index)
    temp["question"]=sample["question"]
    temp["answer"]=sample["answer_name"]
    # find the head entity
    head=[]
    for s in origin:
        if s["question"]==sample["question"]:
            for n in s["graph_query"]["nodes"]:
                if n["node_type"]=="entity":
                    head.append(n["friendly_name"])
            break
    temp["q_entity"]=head
    temp["a_entity"]=sample["answer_name"]
    temp["graph"]=sample["graph_extend_name"]
    temp["choices"]=[]
    process.append(temp)
    index+=1
    
os.makedirs('GNN/grailqa',exist_ok=True)
with open('GNN/grailqa/test.jsonl', "w", encoding="utf-8") as f:
    for item in process:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
 