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

from datasets import load_dataset

dataset = load_dataset('emotion', cache_dir=cache_dir)
classifier = pipeline("zero-shot-classification",device=4)

import torch
from tqdm import tqdm


emotions=['sadness','joy','love','anger','fear','surprise']
reverse_emotions = {emotions[i]:i for i in range(len(emotions))}

labels = dataset['test']['label']

templates = [' i am feeling {}. ',
    '#feeling{}',
    'It\'s a {} feeling']

#------------------- DSS ----------------------------
tidx = 0

print ('-'*20,templates[tidx],'DSS','-'*20)
pred_emos, true_emos = [],[]
for i,line in enumerate(tqdm(dataset['test']['text'])):
    pred_emos.append(classifier(line, emotions, hypothesis_template = templates[tidx])['labels'][0])
    true_emos.append(emotions[labels[i]])

pred_emos_labelled = [reverse_emotions[i] for i in pred_emos]
true_emos_labelled = [reverse_emotions[i] for i in true_emos]
print (classification_report(true_emos_labelled, pred_emos_labelled, target_names=emotions))


#------------------- ESS ----------------------------

manual_emos_mapped = {
        'sadness': ['guilty', 'tired', 'bad', 'sick', 'sad', 'lonely', 'terrible', 'helpless', 'miserable', 'awkward', 'depressed'],
        'fear': ['nervous', 'scared', 'uncomfortable', 'threatened', 'uncertain', 'shaken', 'terrified'],
        'anger': ['angry', 'furious', 'frustrated', 'insulted', 'jealous','mad', 'pissed'],
        'joy': ['good', 'better', 'great', 'happy', 'fine', 'optimistic', 'excited', 'fantastic'],
        'surprise': ['surprised', 'amazed', 'astonished','shocked','stunned', 'impressed','overwhelmed'],
        'love': ['loved','blessed','romantic','sexy','passionate','admired','compassion','affection']
    }

manual_emos_mapped_rev = {}
for k,v in manual_emos_mapped.items():
    for _v in v: manual_emos_mapped_rev[_v]=k

labels = dataset['test']['label']
emotions_all = sum(manual_emos_mapped.values(),[])


tidx = 0

print ('-'*20,templates[tidx],'ESS','-'*20)
pred_emos, true_emos = [],[]
for i,line in enumerate(tqdm(dataset['test']['text'])):
    pred_emos.append(classifier(line, emotions_all, hypothesis_template = templates[tidx])['labels'][0])
    true_emos.append(emotions[labels[i]])


pred_emos_labelled = [reverse_emotions[manual_emos_mapped_rev[i]] for i in pred_emos]
true_emos_labelled = [reverse_emotions[i] for i in true_emos]
# print (true_emos_labelled, pred_emos_labelled)

print (classification_report(true_emos_labelled, pred_emos_labelled, target_names=emotions))