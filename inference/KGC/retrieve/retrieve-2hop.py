import os
import re
import json
import pickle
import random
from tqdm import tqdm
random.seed(43)

DATA = "wikidata5m_transductive"

with open('prompt/base.txt','r',encoding='utf-8') as f:
    base_prompt = f.read()
with open('prompt/kgc_input.txt','r',encoding='utf-8') as f:
    kgc_input_prompt = f.read()
with open('prompt/kgc_output.txt','r',encoding='utf-8') as f:
    kgc_output_prompt = f.read()

endict = pickle.load(open('endict.pkl','rb'))
redict = pickle.load(open('redict.pkl','rb'))
out_tri = pickle.load(open("out_triple.pkl", "rb"))
in_tri  = pickle.load(open("in_triple.pkl", "rb"))
out_en_re = pickle.load(open("out_en_re.pkl", "rb"))
in_en_re  = pickle.load(open("in_en_re.pkl","rb"))

def sort_triples(triples):
    """triples: iterable of (h, r, t) using QIDs
    Return a deterministically sorted list."""
    return sorted(triples, key=lambda x: (x[0], x[1], x[2]))

def get_1hop(e):
    """Return a *sorted list* of (h, r, t) triples around entity e."""
    triples = []
    if out_tri.get(e):
        for r in out_tri[e]:
            triples.append((e, r[0], r[1]))
    if in_tri.get(e):
        for r in in_tri[e]:
            triples.append((r[1], r[0], e))
    triples = list({(h, r, t) for (h, r, t) in triples})
    return sort_triples(triples)

num = 0
tail_data = []
head_data = []

with open(f'../wikidata5m/{DATA}/{DATA}_test.txt','r',encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        num += 1
        line = line.strip().split('\t')
        if endict[line[0]]['label'] is None or redict[line[1]]['label'] is None or endict[line[2]]['label'] is None:
            continue

        # ---------- tail entity prediction ----------
        sub = set()
        e_1hop = set()

        triple1 = get_1hop(line[0])
        for i in triple1:
            # remove ground truth
            if i[0] != line[0] or i[1] != line[1]:
                sub.add(i)
                e_1hop.add(i[0]); e_1hop.add(i[2])

        if len(sub) < 30:
            for e in sorted(e_1hop):
                triple2 = get_1hop(e)  
                for i in triple2:
                    if (i[0] != line[0] or i[1] != line[1]) and len(sub) < 30:
                        sub.add(i)
                        if len(sub) >= 30:
                            break
                if len(sub) >= 30:
                    break

        sub = sorted(sub, key=lambda x: (x[0], x[1], x[2]))[:30]

        enset = set()
        sub_name = []
        for i in sub:
            h, r, t = i
            if endict.get(h) and endict[h]['label'] is not None and \
               redict.get(r) and redict[r]['label'] is not None and \
               endict.get(t) and endict[t]['label'] is not None:
                hname = endict[h]['label']; rname = redict[r]['label']; tname = endict[t]['label']
                sub_name.append((hname, rname, tname))
                enset.add(hname); enset.add(tname)

        prompt = base_prompt + '\n'
        for e in sorted(enset):
            prompt += f'graph.add_node("{e}")\n'
        for hname, rname, tname in sorted(sub_name, key=lambda x: (x[0], x[1], x[2])):
            prompt += f'graph.add_edge("{hname}", "{tname}", relation="{rname}")\n'

        mask_triple = f'({endict[line[0]]["label"]}, {redict[line[1]]["label"]}, [MASK])'
        sample = dict()
        sample["input"] = prompt + kgc_input_prompt.format(triple=mask_triple)
        sample["triple"] = '|'.join(line)
        sample["triple_name"] = endict[line[0]]['label']+'|'+redict[line[1]]['label']+'|'+endict[line[2]]['label']
        sample["mask_triple"] = mask_triple
        sample["gt"] = line[2]
        sample["gt_name"] = endict[line[2]]['label']
        sample["graph_qid"] = sub
        sample["graph_name"] = sub_name
        sample["type"] = "tail"
        tail_data.append(sample)

        # ---------- head entity prediction ----------
        sub = set()
        e_1hop = set()

        triple1 = get_1hop(line[2])
        for i in triple1:
            if i[2] != line[2] or i[1] != line[1]:
                sub.add(i)
                e_1hop.add(i[0]); e_1hop.add(i[2])

        if len(sub) < 30:
            for e in sorted(e_1hop):
                triple2 = get_1hop(e)
                for i in triple2:
                    if (i[2] != line[2] or i[1] != line[1]) and len(sub) < 30:
                        sub.add(i)
                        if len(sub) >= 30:
                            break
                if len(sub) >= 30:
                    break

        sub = sorted(sub, key=lambda x: (x[0], x[1], x[2]))[:30]

        enset = set()
        sub_name = []
        for i in sub:
            h, r, t = i
            if endict.get(h) and endict[h]['label'] is not None and \
               redict.get(r) and redict[r]['label'] is not None and \
               endict.get(t) and endict[t]['label'] is not None:
                hname = endict[h]['label']; rname = redict[r]['label']; tname = endict[t]['label']
                sub_name.append((hname, rname, tname))
                enset.add(hname); enset.add(tname)

        prompt = base_prompt + '\n'
        for e in sorted(enset):
            prompt += f'graph.add_node("{e}")\n'
        for hname, rname, tname in sorted(sub_name, key=lambda x: (x[0], x[1], x[2])):
            prompt += f'graph.add_edge("{hname}", "{tname}", relation="{rname}")\n'

        mask_triple = f'([MASK], {redict[line[1]]["label"]}, {endict[line[2]]["label"]})'
        sample = dict()
        sample["input"] = prompt + kgc_input_prompt.format(triple=mask_triple)
        sample["triple"] = '|'.join(line)
        sample["triple_name"] = endict[line[0]]['label']+'|'+redict[line[1]]['label']+'|'+endict[line[2]]['label']
        sample["mask_triple"] = mask_triple
        sample["gt"] = line[0]
        sample["gt_name"] = endict[line[0]]['label']
        sample["graph_qid"] = sub
        sample["graph_name"] = sub_name
        sample["type"] = "head"
        head_data.append(sample)

os.makedirs(f'../graph/{DATA}', exist_ok=True)
json.dump(tail_data, open(f'../graph/{DATA}/tail.json','w',encoding='utf-8'),
          indent=2, ensure_ascii=False, sort_keys=True)
print(len(tail_data))
print(num)
print(len(head_data))
json.dump(head_data, open(f'../graph/{DATA}/head.json','w',encoding='utf-8'),
          indent=2, ensure_ascii=False, sort_keys=True)

all_data = tail_data + head_data
random.shuffle(all_data)
json.dump(all_data, open(f'../graph/{DATA}/test.json','w',encoding='utf-8'),
          indent=2, ensure_ascii=False, sort_keys=True)
