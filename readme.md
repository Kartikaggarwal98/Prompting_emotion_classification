# Emotion classification using Prompting

Prompting is used to classify a sentence into a label category without any training or finetuning. In this repo, I use two strategy to classify:

1. Direct search space: Predicted words are labels
1. Emotion Search space: Predicted words are a category of labels. Ex: anger: frustrated, mad

Models used:

1. BERT-based models: ```classify_direct.py``` (DSS), ```classify_mapped.py``` (ESS)
1. BART-model: ```bart.py```
1. GPT-model: ```gpt.py```

The predictions for the best model (BERT with prompt "i am feeling" before the sentence) are shown in outs.csv file. Accuracy: 56%, F1: 54%