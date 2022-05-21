cache_dir = '/data/users/kartik/hfcache/'

import os
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

import pandas as pd
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import classification_report

import random
random.seed(123)

dataset = load_dataset('emotion', cache_dir=cache_dir)

print (dataset.column_names)
print (len(dataset['train']), len(dataset['validation']), len(dataset['test']))


emotions=['sadness','joy','love','anger','fear','surprise']
reverse_emotions = {emotions[i]:i for i in range(len(emotions))}

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


#fill mask pipeline directly gives the mask word
model_checkpoint = ['bert-base-cased','bert-base-uncased','distilbert-base-cased','distilbert-base-uncased',
                    'bert-large-cased'][0]

print ('-'*10,model_checkpoint,'-'*10)

unmasker = pipeline('fill-mask', model=model_checkpoint, targets = sum(manual_emos_mapped.values(),[]))

templates = ['i am feeling [MASK]. ',
    'I feel very [MASK]. ',
    '#feeling[MASK]',
    'It\'s a [MASK] feeling',]


true_emos, pred_emos, prompt_sents = [],[],[]

mode = ['train','test','validation'][1]
promptmode = ['pre','post'][0]

tidx = 0
print (f'----------- {templates[tidx]} + {promptmode} + {model_checkpoint} -----------')

for i in tqdm(range(len(dataset[mode]))):
# for i in tqdm(range(10)):
    if promptmode=='pre':
        sent = templates[tidx] + dataset[mode][i]['text']+'.'
    else:
        sent = dataset[mode][i]['text']+ ' '+ templates[tidx]

    prompt_sents.append(sent)

    emo = dataset[mode][i]['label']
    true_emos.append(emotions[emo])

    pred = unmasker(sent)[0]['token_str']
    pred_emos.append(pred)
    
pred_emos_mapped = [manual_emos_mapped_rev[i] for i in pred_emos]

pd.DataFrame(zip(prompt_sents,true_emos,pred_emos,pred_emos_mapped), columns = ['prompt','true','pred','mapped']).to_csv('out.csv',index=None)

pred_emos_labelled = [reverse_emotions[manual_emos_mapped_rev[i]] for i in pred_emos]
true_emos_labelled = [reverse_emotions[i] for i in true_emos]
# print (true_emos_labelled, pred_emos_labelled)

print (classification_report(true_emos_labelled, pred_emos_labelled, target_names=emotions))