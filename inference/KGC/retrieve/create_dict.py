import pickle

# 1hop dict
out_tri=dict()
in_tri=dict()
out_en_re=dict()
in_en_re=dict()
with open('../wikidata5m/wikidata5m_all_triplet.txt', "r", encoding="utf-8") as f:
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

# save dict
pickle.dump(out_tri,open('out_triple.pkl','wb'))
pickle.dump(in_tri,open('in_triple.pkl','wb'))
pickle.dump(out_en_re,open('out_en_re.pkl','wb'))
pickle.dump(in_en_re,open('in_en_re.pkl','wb')) 

pickle.dump(out_tri,open('../infer/out_triple.pkl','wb'))
pickle.dump(in_tri,open('../infer/in_triple.pkl','wb'))
pickle.dump(out_en_re,open('../infer/out_en_re.pkl','wb'))
pickle.dump(in_en_re,open('../infer/in_en_re.pkl','wb'))
        