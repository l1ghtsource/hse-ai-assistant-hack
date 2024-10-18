import numpy as np
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
tqdm.pandas()

model_name = 'DeepPavlov/rubert-base-cased-sentence'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    embedding_str = ' '.join(map(str, embedding))
    return embedding_str


with open('hints_synthlora_qwen32b_140steps.pkl', 'rb') as f:
    hints = pickle.load(f)

sample = pd.read_csv('../data/submit_example.csv')
sample['author_comment'] = hints
sample['author_comment_embedding'] = sample['author_comment'].progress_apply(get_sentence_embedding)
sample.to_csv('hints_synthlora_qwen32b_140steps.csv', index=False)
