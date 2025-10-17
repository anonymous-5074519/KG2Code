import os
import json
import random

random.seed(42)
data1=json.load(open('../pretrain/kgqa/kgqa_corpus.json','r',encoding='utf-8'))
data2=json.load(open('../pretrain/kgc/kgc_corpus.json','r',encoding='utf-8'))

random.shuffle(data1)
random.shuffle(data2)

corpus=[]
corpus.extend(data1)
corpus.extend(data2)

random.shuffle(corpus)

train_num=int(len(corpus)*0.9)

os.makedirs('pretrain/train',exist_ok=True)
json.dump(corpus,open('pretrain/all.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(corpus[:train_num],open('pretrain/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(corpus[train_num:],open('pretrain/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
print(len(corpus))
print(train_num)