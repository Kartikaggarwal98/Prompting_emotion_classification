from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import classification_report

import random
random.seed(123)

cache_dir = '/data/users/kartik/hfcache/'
dataset = load_dataset('emotion', cache_dir=cache_dir)

print (dataset.column_names)
print (len(dataset['train']), len(dataset['validation']), len(dataset['test']))


emotions=['sadness','joy','love','anger','fear','surprise']
reverse_emotions = {emotions[i]:i for i in range(len(emotions))}

model_checkpoint = ['bert-base-cased','bert-base-uncased','distilbert-base-cased','distilbert-base-uncased',
                    'bert-large-uncased'][0]

#fill mask pipeline directly gives the mask word
unmasker = pipeline('fill-mask', model=model_checkpoint, targets = emotions)

templates = ['i am feeling [MASK].',
    '#[MASK]']

true_emos, pred_emos, prompt_sents = [],[],[]

mode = ['train','test','validation'][1]

for i in tqdm(range(len(dataset[mode]))):
    sent = dataset[mode][i]['text']+'. '+ templates[1]
    prompt_sents.append(sent)

    emo = dataset[mode][i]['label']
    true_emos.append(emotions[emo])

    pred = unmasker(sent)[0]['token_str']
    pred_emos.append(pred)
    
pred_emos_labelled = [reverse_emotions[i] for i in pred_emos]
true_emos_labelled = [reverse_emotions[i] for i in true_emos]

print (classification_report(true_emos_labelled, pred_emos_labelled, target_names=emotions))