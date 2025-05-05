# semantic-role-labeling

Implement a program that uses semantic role labeling to extract ARG0 (agent) roles from predicates and identify the 10 most frequent predicate-agent pairs. Use these two approaches:

1. Using the [AllenNLP SRL predictor](https://github.com/allenai/allennlp-models?tab=readme-ov-file#)
2. Using [a fine-tuned pre-trained transformer via Hugging Face](https://huggingface.co/mbruton/spa_en_mBERT)

## Output

- AllenNLP SRL predictor: The 10 most frequency predicate-arg0 pairs:  [(('read', 'I'), 533), (('enjoyed', 'I'), 312), (('reading', 'I'), 250), (('loved', 'I'), 223), (('love', 'I'), 184), (('recommend', 'I'), 170), (('liked', 'I'), 160), (('like', 'I'), 143), (('put', 'I'), 136), (('found', 'I'), 87)]

- Fine-tuned pre-trained transformer: The 10 most frequency predicate-arg0 pairs:  [(('read', 'I'), 348), (('enjoyed', 'I'), 300), (('loved', 'I'), 214), (('love', 'I'), 176), (('recommend', 'I'), 153), (('liked', 'I'), 145), (('like', 'I'), 139), (('put', 'I'), 112), (('reading', 'I'), 86), (('found', 'I'), 73)]

## Data:

sub_amazon-sentences.txt: A subset of 10,000 sentences extracted from the amazon-sentences.txt corpus

## Packages used:

allennlp_models, torch, transformers, re
