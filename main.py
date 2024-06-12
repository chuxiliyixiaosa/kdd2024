#%%
import pandas as pd
import torch
import numpy as np
import os

path='data/'
os.makedirs(path+'models',exist_ok=True)
os.makedirs(path+'feature',exist_ok=True)
os.makedirs(path+'feature_importance',exist_ok=True)
os.makedirs(path+'submissions',exist_ok=True)

version='v002'
#%%
import json

# 打开并读取JSON文件
with open(path+'AQA-test-public/pid_to_title_abs_update_filter.json', 'r', encoding='utf-8') as file:
    pid_to_title_abs_new = json.load(file)

#%%
l1=[i.keys() for i in pid_to_title_abs_new.values()]
tmp_df=pd.DataFrame(l1)
tmp_df.drop_duplicates(subset=None, keep='first', inplace=False)
#%%
pid=list(pid_to_title_abs_new.keys())
pid_to_title_abs_new=[i.values() for i in pid_to_title_abs_new.values()]
pid_to_title_abs_new=pd.DataFrame(pid_to_title_abs_new)
pid_to_title_abs_new.columns=['title','abstract']
pid_to_title_abs_new['pid']=pid
pid_to_title_abs_new['title']=pid_to_title_abs_new['title'].fillna('')
pid_to_title_abs_new['title_abstract']=pid_to_title_abs_new['title']+pid_to_title_abs_new['abstract']
pid_to_title_abs_new
#%%
import json
qa_train=[]
with open(path+'qa_train.txt',encoding='utf-8') as f:
    for line in f.readlines():
        data=json.loads(line)
        qa_train.append(data)
#%%
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'
#%%
qa_train=pd.DataFrame(qa_train)
qa_train['question_body']=qa_train['question'].fillna('')+qa_train['body']
qa_train['pids_len']=qa_train['pids'].apply(lambda x:len(x))
import re

def remove_tags(text):
    """从给定的文本中删除所有尖括号之间的内容。"""
    clean_text = re.sub('<[^>]*>', '', text)
    return clean_text

task = 'Given a question, retrieve Wikipedia passages that answer the question'
qa_train['question_body']=qa_train['question_body'].apply(lambda x:remove_tags(x))
qa_train['question_body']=qa_train['question_body'].apply(lambda x:get_detailed_instruct(task, x))
qa_train
#%%
qa_train_split_df=pd.DataFrame()
l1=[]
for i in qa_train['pids']:
    l1+=i
qa_train_split_df['pid']=l1
l1=[]
for idx,i in enumerate(qa_train['pids']):
    l1+=[qa_train['question_body'][idx]]*len(i)
qa_train_split_df['question_body']=l1
qa_train_split_df=qa_train_split_df.merge(pid_to_title_abs_new[['pid','title_abstract']],how='left',on='pid')
qa_train_split_df
#%%
qa_train_split_df['question_body'][0]
#%%
import json
qa_valid_wo_ans=[]
with open(path+'AQA-test-public/qa_test_wo_ans_new.txt',encoding='utf-8') as f:
    for line in f.readlines():
        data=json.loads(line)
        qa_valid_wo_ans.append(data)
#%%
qa_valid_wo_ans=pd.DataFrame(qa_valid_wo_ans)
qa_valid_wo_ans['question_body']=qa_valid_wo_ans['question'].fillna('')+qa_valid_wo_ans['body']
qa_valid_wo_ans['question_body']=qa_valid_wo_ans['question_body'].apply(lambda x:remove_tags(x))
qa_valid_wo_ans['question_body']=qa_valid_wo_ans['question_body'].apply(lambda x:get_detailed_instruct(task, x))
qa_valid_wo_ans
#%%
qa_valid_wo_ans['question'][0]
#%%
qa_valid_wo_ans['question_body'][0]
#%%
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('Linq-Embed-Mistral')
model = AutoModel.from_pretrained('Linq-Embed-Mistral',torch_dtype=torch.float16,device_map='cuda:0')
#%%
model.dtype,model.device
#%%
import torch
from tqdm import tqdm
if not os.path.exists(path+f'feature/abstract_embeddings_{version}.pt'):
    max_length = 4096
    all_input_texts = pid_to_title_abs_new['title_abstract'].tolist()
    bs=4
    abstract_embeddings=[]
    for i in tqdm(range(0,len(all_input_texts),bs)):
        input_texts=all_input_texts[i:i+bs]
        # Tokenize the input texts
        batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        batch_dict.to(model.device)
        with torch.no_grad():
            outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        abstract_embeddings.append(embeddings.cpu())
    abstract_embeddings=torch.cat(abstract_embeddings,dim=0)
    torch.save(abstract_embeddings,path+f'feature/abstract_embeddings_{version}.pt')
    abstract_embeddings=torch.nn.functional.normalize(abstract_embeddings, p=2, dim=1)
else:
    abstract_embeddings=torch.load(path+f'feature/abstract_embeddings_{version}.pt')
    abstract_embeddings=torch.nn.functional.normalize(abstract_embeddings, p=2, dim=1)
#%%
import torch
from tqdm import tqdm
if not os.path.exists(path+f'feature/question_embeddings_train_{version}.pt'):
    max_length = 4096
    all_input_texts = qa_train['question_body'].tolist()
    bs=4
    question_embeddings_train=[]
    for i in tqdm(range(0,len(all_input_texts),bs)):
        input_texts=all_input_texts[i:i+bs]
        # Tokenize the input texts
        batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        batch_dict.to(model.device)
        with torch.no_grad():
            outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        question_embeddings_train.append(embeddings.cpu())
    question_embeddings_train=torch.cat(question_embeddings_train,dim=0)
    torch.save(question_embeddings_train,path+f'feature/question_embeddings_train_{version}.pt')
    question_embeddings_train=torch.nn.functional.normalize(question_embeddings_train, p=2, dim=1)
else:
    question_embeddings_train=torch.load(path+f'feature/question_embeddings_train_{version}.pt')
    question_embeddings_train=torch.nn.functional.normalize(question_embeddings_train, p=2, dim=1)
#%%
import torch
from tqdm import tqdm
if not os.path.exists(path+f'feature/question_embeddings_test_{version}.pt'):
    max_length = 4096
    all_input_texts = qa_valid_wo_ans['question_body'].tolist()
    bs=4
    question_embeddings_test=[]
    for i in tqdm(range(0,len(all_input_texts),bs)):
        input_texts=all_input_texts[i:i+bs]
        # Tokenize the input texts
        batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        batch_dict.to(model.device)
        with torch.no_grad():
            outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        question_embeddings_test.append(embeddings.cpu())
    question_embeddings_test=torch.cat(question_embeddings_test,dim=0)
    torch.save(question_embeddings_test,path+f'feature/question_embeddings_test_{version}.pt')
    question_embeddings_test=torch.nn.functional.normalize(question_embeddings_test, p=2, dim=1)
else:
    question_embeddings_test=torch.load(path+f'feature/question_embeddings_test_{version}.pt')
    question_embeddings_test=torch.nn.functional.normalize(question_embeddings_test, p=2, dim=1)
#%%

def cos_similarity_matrix(a: torch.Tensor, b: torch.Tensor):
    """Calculates cosine similarities between tensor a and b."""

    sim_mt = torch.mm(a, b.transpose(0, 1)) # 就是矩阵乘法 用的是这个公式 cos(θ) = (A · B) / (||A|| ||B||) 我之前除以单位向量再算欧式距离其实是等价方法并不是直接计算
    return sim_mt


def get_topk(embeddings_from, embeddings_to, topk=1000, bs=512): # from是查询向量，to是被查询向量
    chunk = bs
    embeddings_chunks = embeddings_from.split(chunk) #把查询训练按batch_size拆分

    vals = []
    inds = []
    for idx in range(len(embeddings_chunks)):
        cos_sim_chunk = cos_similarity_matrix(
            embeddings_chunks[idx].to(embeddings_to.device).half(), embeddings_to.half()
        ).float() # 相似度转fp32

        cos_sim_chunk = torch.nan_to_num(cos_sim_chunk, nan=0.0) #把nan处理成0

        topk = min(topk, cos_sim_chunk.size(1))
        vals_chunk, inds_chunk = torch.topk(cos_sim_chunk, k=topk, dim=1)
        vals.append(vals_chunk[:, :].detach().cpu())
        inds.append(inds_chunk[:, :].detach().cpu())

        del vals_chunk
        del inds_chunk
        del cos_sim_chunk

    vals = torch.cat(vals).detach().cpu()
    inds = torch.cat(inds).detach().cpu()

    return inds, vals


#%%
inds, vals=get_topk(question_embeddings_train.cuda(),abstract_embeddings.cuda(),20,bs=32)
infer_pids=pid_to_title_abs_new['pid'].values[inds]
ground_truth=np.array([[1 if i in qa_train['pids'][idx] else 0 for i in l1] for idx,l1 in enumerate(infer_pids)])
weights=np.array([[1/(i+1) for i in range(20)]]*len(ground_truth))
(ground_truth*weights).sum(axis=1).mean()
#%%

#%%

#%%
inds, vals=get_topk(question_embeddings_test.cuda(),abstract_embeddings.cuda(),20,bs=32)
infer_pids=pid_to_title_abs_new['pid'].values[inds]
infer_pids
#%%
import time
infer_df=pd.DataFrame(infer_pids)
infer_df.to_csv('submissions/results.txt', index=False,header=None)
infer_df
#%%
