import pickle
import os

DATA='WN18RR'

# entity dict
endict=dict()
if DATA!='WN18RR':
    with open('dataset/'+DATA+'/entity2text.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            line=line.strip().split('\t')
            endict[line[0]]=line[1]
else:
    with open('dataset/'+DATA+'/wordnet-mlj12-definitions.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            line=line.strip().split('\t')
            name=' '.join(line[1].strip('_').split('_'))
            endict[line[0]]=name    

# relation dict
redict=dict()
with open('dataset/'+DATA+'/relation2text.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        line=line.strip().split('\t')
        redict[line[0]]=line[1]

# 1hop dict
out_tri=dict()
in_tri=dict()
out_en_re=dict()
in_en_re=dict()
with open('dataset/'+DATA+"/train.tsv", "r", encoding="utf-8") as f:
    for line in f:
        row = line.strip().split("\t")
        if out_tri.get(row[0]) is None:
            out_tri[row[0]]=set()
        out_tri[row[0]].add((row[1],row[2]))
        if in_tri.get(row[2]) is None:
            in_tri[row[2]]=set()
        in_tri[row[2]].add((row[1],row[0]))
        if out_en_re.get((row[0],row[1])) is None:
            out_en_re[(row[0],row[1])]=set()
        out_en_re[(row[0],row[1])].add(row[2])
        if in_en_re.get((row[2],row[1])) is None:
            in_en_re[(row[2],row[1])]=set() 
        in_en_re[(row[2],row[1])].add(row[0])
        
with open('dataset/'+DATA+"/dev.tsv", "r", encoding="utf-8") as f:
    for line in f:
        row = line.strip().split("\t")
        if out_tri.get(row[0]) is None:
            out_tri[row[0]]=set()
        out_tri[row[0]].add((row[1],row[2]))
        if in_tri.get(row[2]) is None:
            in_tri[row[2]]=set()
        in_tri[row[2]].add((row[1],row[0]))
        if out_en_re.get((row[0],row[1])) is None:
            out_en_re[(row[0],row[1])]=set()
        out_en_re[(row[0],row[1])].add(row[2])
        if in_en_re.get((row[2],row[1])) is None:
            in_en_re[(row[2],row[1])]=set() 
        in_en_re[(row[2],row[1])].add(row[0])
        
with open('dataset/'+DATA+"/test.tsv", "r", encoding="utf-8") as f:
    for line in f:
        row = line.strip().split("\t")
        if out_tri.get(row[0]) is None:
            out_tri[row[0]]=set()
        out_tri[row[0]].add((row[1],row[2]))
        if in_tri.get(row[2]) is None:
            in_tri[row[2]]=set()
        in_tri[row[2]].add((row[1],row[0]))
        if out_en_re.get((row[0],row[1])) is None:
            out_en_re[(row[0],row[1])]=set()
        out_en_re[(row[0],row[1])].add(row[2])
        if in_en_re.get((row[2],row[1])) is None:
            in_en_re[(row[2],row[1])]=set() 
        in_en_re[(row[2],row[1])].add(row[0])

os.makedirs('graph/'+DATA, exist_ok=True)
# save dict
pickle.dump(endict,open('graph/'+DATA+'/endict.pkl','wb'))
pickle.dump(redict,open('graph/'+DATA+'/redict.pkl','wb'))
pickle.dump(out_tri,open('graph/'+DATA+'/out_triple.pkl','wb'))
pickle.dump(in_tri,open('graph/'+DATA+'/in_triple.pkl','wb'))
pickle.dump(out_en_re,open('graph/'+DATA+'/out_en_re.pkl','wb'))
pickle.dump(in_en_re,open('graph/'+DATA+'/in_en_re.pkl','wb'))