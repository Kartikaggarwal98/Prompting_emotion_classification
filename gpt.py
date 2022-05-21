cache_dir = '/data/users/kartik/hfcache/'

import os
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir

import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import classification_report

import random
random.seed(123)

device = 'cuda:4'

from datasets import load_dataset

dataset = load_dataset('emotion', cache_dir=cache_dir)

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
model_checkpoint = 'gpt2'
tokenizer = GPT2TokenizerFast.from_pretrained(model_checkpoint,cache_dir=cache_dir) #gpt2-xl, EleutherAI/gpt-j-6B
model = GPT2LMHeadModel.from_pretrained(model_checkpoint,cache_dir=cache_dir).to(device)

import torch
from tqdm import tqdm

print ('------------------- DSS ----------------------------')

sum_len = 0
nlls = []
emotions=['sadness','joy','love','anger','fear','surprise']
reverse_emotions = {emotions[i]:i for i in range(len(emotions))}

labels = dataset['test']['label']
model = model.to(device)

pred_emos, true_emos = [],[]
for i,line in enumerate(tqdm(dataset['test']['text'])):
    best_emo, best_val = None, float('inf')

    for emo in emotions:
        _line = line+f'. I am feeling {emo}.'

        ids = tokenizer.encode(_line)
        input_ids = torch.tensor(ids).to(device)
        target_ids = input_ids.clone().to(device)
        trg_len = len(ids)
        sum_len += trg_len

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        if neg_log_likelihood < best_val:
            best_emo, best_val = emo, neg_log_likelihood

    pred_emos.append(best_emo)
    true_emos.append(emotions[labels[i]])


pred_emos_labelled = [reverse_emotions[i] for i in pred_emos]
true_emos_labelled = [reverse_emotions[i] for i in true_emos]
print (classification_report(true_emos_labelled, pred_emos_labelled, target_names=emotions))

print ('------------------- ESS ----------------------------')

sum_len = 0
nlls = []
emotions=['sadness','joy','love','anger','fear','surprise']
reverse_emotions = {emotions[i]:i for i in range(len(emotions))}

labels = dataset['test']['label']
model = model.to(device)

manual_emos_mapped = {
        'sadness': ['guilty', 'tired', 'bad', 'sick', 'sad', 'lonely', 'terrible', 'helpless', 'miserable', 'awkward', 'depressed'],
        'fear': ['nervous', 'scared', 'weird', 'uncomfortable', 'threatened', 'vulnerable', 'uncertain', 'shaken', 'terrified', 'alone', 'free', 'tortured', 'frightened', 'trapped', 'fearful'],
        'anger': ['angry', 'furious', 'frustrated', 'insulted', 'jealous', 'impatient', 'greedy', 'bitter', 'mad', 'pissed', 'restless', 'rebellious'],
        'joy': ['good', 'better', 'great', 'happy', 'proud', 'confident', 'fine', 'optimistic', 'rich', 'brave', 'satisfied', 'alive', 'excited', 'lucky', 'relaxed', 'beautiful', 'amazing', 'pretty', 'wonderful', 'free', 'honoured', 'playful', 'fantastic'],
        'surprise': ['surprised', 'amazed', 'astonished','shocked','stunned', 'impressed','overwhelmed'],
        'love': ['loved','blessed','generous','romantic','sexy','passionate','admired','compassion','sympathetic','affection']
    }

manual_emos_mapped_rev = {}
for k,v in manual_emos_mapped.items():
    for _v in v: manual_emos_mapped_rev[_v]=k


pred_emos, true_emos = [],[]
for i,line in enumerate(tqdm(dataset['validation']['text'])):
    best_emo, best_val = None, float('inf')

    for emo in sum(manual_emos_mapped.values(),[]):
        _line = line+f'. I am feeling {emo}.'

        ids = tokenizer.encode(_line)
        input_ids = torch.tensor(ids).to(device)
        target_ids = input_ids.clone().to(device)
        trg_len = len(ids)
        sum_len += trg_len

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        if neg_log_likelihood < best_val:
            best_emo, best_val = emo, neg_log_likelihood

    # print (line, best_emo, best_val, emotions[labels[i]])
    pred_emos.append(best_emo)
    true_emos.append(emotions[labels[i]])

