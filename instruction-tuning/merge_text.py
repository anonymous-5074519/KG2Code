import os
import json
import random

random.seed(42)
data1=json.load(open('../pretrain/kgqa/kgqa_text_corpus.json','r',encoding='utf-8'))
data2=json.load(open('../pretrain/kgc/kgc_text_corpus.json','r',encoding='utf-8'))

random.shuffle(data1)
random.shuffle(data2)

corpus=[]
corpus.extend(data1)
corpus.extend(data2)

random.shuffle(corpus)

train_num=int(len(corpus)*0.9)

os.makedirs('pretrain_text/train',exist_ok=True)
json.dump(corpus,open('pretrain_text/all.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(corpus[:train_num],open('pretrain_text/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(corpus[train_num:],open('pretrain_text/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
print(len(corpus))
print(train_num)