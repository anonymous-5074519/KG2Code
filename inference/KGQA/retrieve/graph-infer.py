import json
import pickle
from tqdm import tqdm

DATASET='QALD-9'

# wikidata5m endict/redict
with open("endict.pkl", "rb") as file:
    endict = pickle.load(file)

with open("redict.pkl", "rb") as file:
    redict = pickle.load(file)

with open('prompt/kgqa_input.txt','r',encoding='utf-8') as f:
    qa_input_prompt = f.read()
    
with open('prompt/kgqa_output.txt','r',encoding='utf-8') as f:
    qa_output_prompt = f.read()
    
with open('prompt/base.txt','r',encoding='utf-8') as f:
    base_prompt = f.read()

data=json.load(open('../graph/'+DATASET+'/graph.json'))
# deal with dataset
process_data=[]
for sample in tqdm(data):
    graph_name=[]
    # construct the name format of graph_mid
    for g in sample['graph_mid']:
        g_name=[]
        for t in g:
            t_name=[]
            if endict.get(t[0]) and endict[t[0]]['label'] is not None:
                t_name.append(endict[t[0]]['label'])
            else:
                t_name.append(t[0])
            if redict.get(t[1]) and redict[t[1]]['label'] is not None:
                t_name.append(redict[t[1]]['label'])
            else:
                t_name.append(t[1])
            if endict.get(t[2]) and endict[t[2]]['label'] is not None:
                t_name.append(endict[t[2]]['label'])
            else:
                t_name.append(t[2])
            g_name.append(t_name)
        graph_name.append(g_name)
    sample['graph_name']=graph_name 
    graph_extend_name=[]
    # construct the name format of graph_extend_mid
    for t in sample['graph_extend_mid']:
        t_name=[]
        if endict.get(t[0]) and endict[t[0]]['label'] is not None:
            t_name.append(endict[t[0]]['label'])
        else:
            t_name.append(t[0])
        if redict.get(t[1]) and redict[t[1]]['label'] is not None:
            t_name.append(redict[t[1]]['label'])
        else:
            t_name.append(t[1])
        if endict.get(t[2]) and endict[t[2]]['label'] is not None:
            t_name.append(endict[t[2]]['label'])
        else:
            t_name.append(t[2])
        graph_extend_name.append(t_name)
    sample['graph_extend_name']=graph_extend_name
    # construct the name format of answer_mid
    answer_name=[]
    for a in sample["answer_mid"]:
        if endict.get(a) and endict[a]["label"] is not None:
            answer_name.append(endict[a]["label"])
        elif a.lower() in ['true','yes']:
            answer_name.append('yes')
        elif a.lower() in ['false','no']:
            answer_name.append('no')        
        else:
            answer_name.append(a)
    sample["answer_name"]=answer_name
    prompt=base_prompt+'\n'
    # collect all the entity, including literal
    enset=set()
    for t in sample["graph_extend_name"]:
        enset.add(t[0])
        enset.add(t[2])
    # deal with entity
    for e in enset:
        prompt=prompt+'graph.add_node("{name}")'.format(name=e)+'\n'
    # deal with relation
    for t in sample["graph_extend_name"]:
        prompt=prompt+'graph.add_edge("{head}", "{tail}", relation="{relation}")'.format(head=t[0],tail=t[2],relation=t[1])+'\n'
    # kgqa
    sample["input"]=prompt+qa_input_prompt.format(question=sample["question"])
    process_data.append(sample)

# save data after adding graph_extend_name
json.dump(process_data, open('../graph/'+DATASET+'/test.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)