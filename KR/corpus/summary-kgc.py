import os
import json
import time
from openai import OpenAI
from tqdm import tqdm

data=json.load(open('../../pretrain/kgc/kgc_cot.json','r',encoding='utf-8'))

client=OpenAI(api_key='YOUR API KEY')

kr_prompt='''Your task is to summarize the relevant knowledge that is helpful to predict the missing part of the triple from the following subgraph in one line.
Subgraph: (X-Alfonso, country of citizenship, Cuba) (X-Alfonso, genre, jazz fusion) (Habana Blues, composer, X-Alfonso) (X-Alfonso, place of birth, Havana) (X-Alfonso, instance of, human)
Triple: (X-Alfonso, occupation, [MASK])
Knowledge: X-Alfonso is associated with the genre of jazz fusion. X-Alfonso is the composer of "Habana Blues".

Your task is to summarize the relevant knowledge that is helpful to predict the missing part of the triple from the following subgraph in one line.
Subgraph: (Karwacja, located in the administrative territorial entity, Gmina Sierakowice) (Karwacja, country, Poland)
Triple: (Karwacja, located in time zone, [MASK])
Knowledge: Karwacja is located in Poland.

Your task is to summarize the relevant knowledge that is helpful to predict the missing part of the triple from the following subgraph in one line.
Subgraph: (Standard, followed by, Hello World) (Hello World, performer, Scandal) (Hello World, language of work or name, Japanese) (Hello World, instance of, album) (Hello World, followed by, Yellow) (Yellow, follows, Hello World) (Hello World, record label, Epic Records Japan) (Hello World, follows, Standard)
Triple: (Hello World, genre, [MASK])
Knowledge: "Hello World" is an album performed by "Scandal", with lyrics in Japanese. Additionally, it is part of a sequence of works, being followed by "Yellow" and preceded by "Standard".

Your task is to summarize the relevant knowledge that is helpful to predict the missing part of the triple from the following subgraph in one line.
Subgraph: {subgraph}
Triple: {triple}
Knowledge: '''

kr_prompt1='''Your task is to summarize the relevant knowledge that is helpful to predict the missing part of the triple from the following subgraph in one line.
Subgraph: {subgraph}
Triple: {triple}
Knowledge: '''

def getResponse(prompt,max_retries=10):
    # set retries
    retries=0
    while retries < max_retries:
        try:
            res = client.chat.completions.create(
                #model='gpt-3.5-turbo',
                #model='gpt-4o-mini',
                model='gpt-4o',
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0,
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying in 1 minutes...")
            retries += 1
            time.sleep(60)
    return ''

process=[]
for sample in tqdm(data):
    # triples linearization
    triple=''
    for t in sample["subgraph name"]:
        triple=triple+'('+t[0]+', '+t[1]+', '+t[2]+') '
    triple=triple.strip()
    mask='('+', '.join(sample["masked triple name"])+')'
    knowledge=getResponse(kr_prompt.format(subgraph=triple.strip(),triple=mask))
    print(kr_prompt.format(subgraph=triple.strip(),triple=mask))
    print(knowledge)
    
    # record
    sample['know_prompt']=kr_prompt1.format(subgraph=triple.strip(),triple=mask)
    sample['knowledge']=knowledge.strip()
    process.append(sample)
    # generate 2,000 samples
    if len(process)>=2000:
        break
   
os.makedirs("summary", exist_ok=True)   

json.dump(process,open('summary/kgc.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)  
