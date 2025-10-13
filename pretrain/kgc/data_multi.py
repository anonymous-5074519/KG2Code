import json
import pickle
import random
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

random.seed(42)

out_tri = pickle.load(open("out_triple.pkl", "rb"))
in_tri = pickle.load(open("in_triple.pkl", "rb"))
out_en_re = pickle.load(open("out_en_re.pkl", "rb"))
in_en_re = pickle.load(open("in_en_re.pkl","rb"))
endict = pickle.load(open("endict.pkl", "rb"))
redict = pickle.load(open("redict.pkl", "rb"))

with open('entity.txt', 'r', encoding='utf-8') as f:
    enlist = [line.strip() for line in f]

random.shuffle(enlist)

def get_1hop(e):
    triple = set()
    if out_tri.get(e):
        for r in out_tri[e]:
            triple.add((e, r[0], r[1]))
    if in_tri.get(e):
        for r in in_tri[e]:
            triple.add((r[1], r[0], e))
    return triple

def get_2hop(e):
    triple = get_1hop(e)
    enset = {t[0] for t in triple} | {t[2] for t in triple}
    enset.discard(e)
    for i in enset:
        triple |= get_1hop(i)
    return triple

def process_entity(e):
    triple1 = get_1hop(e)
    #triple2 = get_2hop(e)
    if not triple1:
        return None

    mask_triple = random.choice(list(triple1))
    mask_rel=mask_triple[1]
    # remove the same relation from triple1
    if mask_triple[0] == e: 
        triple1 = {t for t in triple1 if not (t[0] == e and t[1] == mask_rel)}
    else:
        triple1 = {t for t in triple1 if not (t[2] == e and t[1] == mask_rel)}
        
    #triple1.remove(mask_triple)
    #triple2.remove(mask_triple)

    subgraph = [list(t) for t in list(triple1)[:30]]
    #if len(subgraph)<3:
    #    return None
    '''
    if len(subgraph) < 30:
        for t in triple2:
            subgraph.append(list(t))
            if len(subgraph) >= 30:
                break
    '''
    # mask tail entity
    if mask_triple[0] == e:
        mask = [e, mask_triple[1], "[MASK]"]
        # consider all plausible answers, only remain top 10 answers
        answer_candidate = list(out_en_re[(e, mask_triple[1])])
        answer=[]
        answer_name=[]
        for a in answer_candidate:
            if endict.get(a) and endict[a]["label"] is not None:
                answer.append(a)
                answer_name.append(endict[a]["label"])
    else:
        mask = ["[MASK]", mask_triple[1], e]
        # consider all plausible answers, only remain top 10 answers
        answer_candidate = list(in_en_re[(e,mask_triple[1])])
        answer=[]
        answer_name=[]
        for a in answer_candidate:
            if endict.get(a) and endict[a]["label"] is not None:
                answer.append(a)
                answer_name.append(endict[a]["label"])
                
    # verify the data
    if mask[-1] == "[MASK]":
        if endict.get(mask[0]) and endict[mask[0]]["label"] is not None and redict.get(mask[1]) and redict[mask[1]]["label"] is not None and len(answer)!=0:
            mask_name = [endict[mask[0]]['label'], redict[mask[1]]['label'], "[MASK]"]
        else:
            return None
    else:
        if endict.get(mask[2]) and endict[mask[2]]["label"] is not None and redict.get(mask[1]) and redict[mask[1]]["label"] is not None and len(answer)!=0:
            mask_name = ["[MASK]", redict[mask[1]]['label'], endict[mask[2]]['label']]
        else:
            return None

    head_name = [endict[e]['label']]

    subgraph_name = []
    for t in subgraph:
        if endict.get(t[0]) and endict[t[0]]["label"] is not None and endict.get(t[2]) and endict[t[2]]["label"] is not None and redict.get(t[1]) and redict[t[1]]["label"] is not None:
            t_name = [endict[t[0]]['label'],redict[t[1]]['label'],endict[t[2]]['label']]
            subgraph_name.append(t_name)

    return {
        "head entity": [e],
        "answer": answer,
        "subgraph": subgraph,
        "head entity name": head_name,
        "answer name": answer_name,
        "subgraph name": subgraph_name,
        "masked triple": mask,
        "masked triple name": mask_name
    }

if __name__ == '__main__':
    with Pool(5) as pool:
        results = list(tqdm(pool.imap(process_entity, enlist), total=len(enlist)))

    corpus = [r for r in results if r is not None]

    with open('kgc-data.json', 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
