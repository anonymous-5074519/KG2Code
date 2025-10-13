import json
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from query import run_sparql

DATASET='QALD-10'

ignore_keywords = {'FILTER','BIND','OPTIONAL','MINUS','ORDER'}  # Use a set for faster lookup

def parse_sparql(query: str) -> List[List[str]]:
    """
    Parses a SPARQL query and extracts the RDF triples while ignoring specified patterns.
    
    Args:
        query (str): The input SPARQL query as a string.
    
    Returns:
        List[List[str]]: A list of extracted triples represented as lists of subject, predicate, and object.
    """
    FLAG=False
    for i in ignore_keywords:
        if i in query:
            key_id=query.index(i)
            if '}' in query[key_id:]:
                query=query.replace(query[key_id:],'')+'}'
            else:
                query=query.replace(query[key_id:],'')
    start_idx = query.index('{')
    end_idx = query.rindex('}')
    content = query[start_idx + 1:end_idx].strip()
    # Normalize whitespace and remove trailing dots/spaces
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'[.\s]+$', '', content)
    
    triples = []
    statements = content.split('. ')
    for statement in statements:
        statement = statement.strip()
        
        # Handle `;` separated statements
        if ';' in statement:
            statement = re.sub(r'\s+;', ';', statement)
            parts = statement.split(' ')
            for index in range(1,len(parts)-2,2):
                triples.append([parts[0], parts[index], parts[index+1][:-1]])  # Remove trailing `;`
            triples.append([parts[0], parts[len(parts)-2], parts[len(parts)-1]])
            continue
        else:
            # Split into triple components
            parts = statement.split(' ')
            if len(parts) == 3:
                triples.append(parts)
    triples=[[s.rstrip("*,;") for s in sublist] for sublist in triples] 
    
    return triples

def normalize(graph):
    gmid = []
    gstr = []
    for g in graph:
        if g[-1][0] == 'Q' and g[-1][1:].isdigit():
            gmid.append(g)
        else:
            gstr.append(g)
    return gmid + gstr

def normalize_triples(graph: List[List[str]]) -> List[List[str]]:
    """
    Normalizes the extracted RDF triples by simplifying URIs and handling prefixes.
    
    Args:
        graph (List[List[str]]): A list of RDF triples.
    
    Returns:
        List[List[str]]: Normalized triples with simplified URIs.
    """
    normalized_graph = []
    
    for triple in graph:
        normalized_triple = []
        
        for element in triple:
            if '<' in element:
                uris = re.findall(r"<(.*?)>", element)
                
                if len(uris) == 1:
                    uri_parts = uris[0].split('/')
                    normalized_value = uri_parts[-2] if uri_parts[-1] == '' else uri_parts[-1]
                    normalized_triple.append(normalized_value)
                else:
                    connector = re.findall(r">(.*?)<", element)[0]
                    simplified_uris = [uri.split('/')[-2] if uri.split('/')[-1] == '' else uri.split('/')[-1] for uri in uris]
                    normalized_triple.append(connector.join(simplified_uris))
                
            elif ':' in element:
                normalized_triple.append(element.split(':')[-1])
            elif element.startswith('?'):
                normalized_triple.append(element)
            else:
                normalized_triple.append(element)
                
        normalized_graph.append(normalized_triple)
    normalized_graph=[[s.rstrip("*,;") for s in sublist] for sublist in normalized_graph]   
    
    return normalized_graph
    
def process_one_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    process_sample = dict()
    query = sample["query"]["sparql"]
    FLAG=False
    query1=query
    for i in ignore_keywords:
        if i in query1:
            key_id=query1.index(i)
            if '}' in query1[key_id:]:
                query1=query1.replace(query1[key_id:],'')+'}'
            else:
                query1=query1.replace(query1[key_id:],'')
    # deal with , in query
    if ',' in query1:
        part_re=re.findall(r'(\S+) [\S]+,', query1)
        for r in part_re:
            query=query.replace(',','; '+r,1)  
    # deal with / in query
    if '/wdt' in query1 or '/<' in query1 or '/ps' in query1:
        part_re = re.findall(r'(/wdt[^/\s]*|/<[^/\s]*|/ps[^/\s]*)', query1)
        for index,r in enumerate(part_re):
            re_sign='?temp'+str(index+1)
            query=query.replace(r,' '+re_sign+' . '+re_sign+' '+r[1:],1)
    parsed_triples = parse_sparql(query)
    normalized_triples = normalize_triples(parsed_triples)
    # format checking
    FLAG=True
    for t in normalized_triples:
        if len(t)!=3:
            FLAG=False
    # ask no question has no subgraph
    if "boolean" in sample["answers"][0] and not sample["answers"][0]["boolean"]:
        FLAG=False
    qset=set()
    for t in normalized_triples:
        for p in t:
            if p.startswith('?'):
                qset.add(p)
    qlist=list(qset)
    if len(qlist)!=0 and FLAG:   
        query = re.sub(r'SELECT [\?A-Za-z\(\)\d ]+{', 'SELECT DISTINCT ' + ' '.join(qlist) + ' WHERE {', query,count=1)
        query = re.sub(r'[Aa][Ss][Kk][\?A-Za-z\(\)\d ]*{', 'SELECT DISTINCT ' + ' '.join(qlist) + ' WHERE {', query,count=1)
        try:
            result=run_sparql(query)
            if len(result['results']['bindings'])==0 or result['results']['bindings']==[{}]:
                FLAG=False
        except:
            FLAG=False
        if FLAG:
            relist=[]
            for r in result['results']['bindings']:
                temp=dict()
                # deal with r
                for i in r.items():
                    if i[1]['value'].startswith('http'):
                        value=i[1]['value'].split('/')[-1]
                        if not value.startswith('Q'):
                            value=i[1]['value']
                    else:
                        value=i[1]['value']
                    temp['?'+i[0]]=value
                relist.append(temp)
            # construct gold subgraph
            gold_mid=[]
            for redict in relist:
                sub_one=[]
                for t in normalized_triples:
                    temp=[]
                    for p in t:
                        if p.startswith('?'):
                            temp.append(redict[p])
                        else:
                            temp.append(p)
                    sub_one.append(temp)
                gold_mid.append(sub_one)
        else:
            gold_mid=[]
    # ask yes question
    elif "boolean" in sample["answers"][0] and sample["answers"][0]["boolean"] and len(qlist)==0 and len(normalized_triples)!=0:
        gold_mid=[normalized_triples]
    else:
        gold_mid=[]
    for q in sample["question"]:
        if q["language"]=="en":
            process_sample["question"]=q["string"]
            break
    
    answer_mid=[]
    if sample["answers"][0].get("results"):
        for a in sample["answers"][0]["results"]["bindings"]:
            temp=list(a.values())[0]["value"]
            if temp.startswith("http"):
                temp=temp.split("/")[-1]
            if temp.startswith('Q'):
                answer_mid.append(temp)
            else:
                answer_mid.append(list(a.values())[0]["value"])
    else:
        answer_mid.append(str(sample["answers"][0]['boolean']))    
    answer_mid=list(set(answer_mid))   
    process_sample["answer_mid"]=answer_mid
    process_sample["graph_mid"]=gold_mid
    enset=set()
    reset=set()
    for g in gold_mid:
        for t in g:
            if t[0][0]=='Q' and t[0][1:].isdigit():
                enset.add(t[0])
            if t[2][0]=='Q' and t[2][1:].isdigit():
                enset.add(t[2])
            reset.add(t[1])
    process_sample["entity"]=list(enset)
    process_sample["relation"]=list(reset)
    process_sample["sparql"]=sample["query"]["sparql"]
    return process_sample
    
def extract_graph_multiprocess(data, max_workers: Optional[int] = None):
    results = []
    total = len(data)
    if max_workers is None:
        max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_sample, sample) for sample in data]
        for f in tqdm(as_completed(futures), total=total, desc="Processing (multiprocess)"):
            res = f.result()
            if res is not None:
                results.append(res)
    return results

if __name__ == "__main__":
    data = json.load(open('../dataset/' + DATASET + '/qald_10.json', 'r', encoding='utf-8'))["questions"]
    processed = extract_graph_multiprocess(data)
    os.makedirs('../graph/' + DATASET,exist_ok=True)
    out_path = '../graph/' + DATASET + '/origin.json'
    json.dump(processed, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"Saved {len(processed)} items to {out_path}")