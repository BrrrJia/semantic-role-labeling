from allennlp_models import pretrained
from collections import Counter, defaultdict
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import Tuple, Optional


tokenizer = AutoTokenizer.from_pretrained("mbruton/spa_en_mBERT")
model = AutoModelForTokenClassification.from_pretrained("mbruton/spa_en_mBERT")

def read_corpus(txt_corpus):
    """
    read in txt corpus file and return as a list of sentences

    Parameters:
        txt_corpus (str): corpus

    Returns:
        sentences (list): modified corpus as a list of sentences
    """
    with open(txt_corpus, 'r') as corpus:
        sentences = corpus.read().split('\n')
       
    return sentences


def extract_pred_arg0_pairs(sentence: str):
    """
    Extract predicate-argument pairs from a sentence using the mBERT model.
    
    Args:
        sentence (str): Input sentence to analyze
        
    Returns:
        Optional[Tuple[str, str]]: A tuple of (predicate, argument) if found, None otherwise
        
    Raises:
        ValueError: If the input sentence is empty or invalid
    """
    if not sentence or not isinstance(sentence, str):
        raise ValueError("Input must be a non-empty string")
        
    # Tokenize input
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Process predictions
    predictions = torch.argmax(outputs.logits, dim=2)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[idx.item()] for idx in predictions]
    
    results = []
    pair_tokens = defaultdict(lambda: [[], []])
    for token, label in zip(tokens, labels):
        if token in tokenizer.all_special_tokens:
            continue
        if ":root" in label:
            pair_tokens[label[1]][0].append(token)
        if ":arg0" in label:
            pair_tokens[label[1]][1].append(token)
    
    for i in pair_tokens.values():
        if i[0] and i[1]:
            pred = tokenizer.convert_tokens_to_string(i[0])
            arg0 = tokenizer.convert_tokens_to_string(i[1])
            results.append((pred, arg0))
    
    return results


def predict_with_allennlp(txt_corpus):
    """
    predict the predicates and their arg0s using allennlp model,
    and then extract the predicate-arg0 pairs from the results
    and sort the pairs according to frequency,
    and return the 10 most frequent pairs

    Parameters:
        txt_corpus (str): corpus

    Returns:
        top_10_results (list): the 10 most frequent predicate-arg0 pairs
    """
    sentences = read_corpus(txt_corpus)
    predictor = pretrained.load_predictor("structured-prediction-srl-bert")
    results = []
    for sent in sentences:
        result = predictor.predict(sentence=sent)
        predicates = result['verbs']
        if predicates == []:
            print(sent)
        for predicate in predicates:
            if 'ARG0' in predicate['description']:
                pred = predicate['verb']
                pattern = r'ARG0:\s*(.*?)\]'
                match = re.search(pattern, predicate['description'])
                arg0 = match.group(1)
                results.append((pred, arg0))
    frequency = dict(Counter(results))
    sorted_frequency = list(sorted(frequency.items(), key=lambda i:i[1], reverse=True))
    top_10_results = sorted_frequency[:10]
    return top_10_results


def predict_with_huggingface(txt_corpus):
    """
    Predict predicates and their arg0s using the HuggingFace model, extract predicate-arg0 pairs,
    and return the 10 most frequent pairs.

    Args:
        txt_corpus (str): Path to the corpus file containing sentences

    Returns:
        top_10_results (list): the 10 most frequent predicate-arg0 pairs
    """
    sentences = read_corpus(txt_corpus)
    results = []
    for sent in sentences:
        pred_arg0_pair = extract_pred_arg0_pairs(sent)
        if pred_arg0_pair:
            results.extend(pred_arg0_pair)
    frequency = dict(Counter(results))
    sorted_frequency = list(sorted(frequency.items(), key=lambda i:i[1], reverse=True))
    top_10_results = sorted_frequency[:10]

    return top_10_results


if __name__ == '__main__':
    # Uncomment to use Allennlp srl predictor
    #print("The 10 most frequency predicate-arg0 pairs: ", predict_with_allennlp('/Data/sub_amazon-sentences.txt'))

    # use hugging-face pre-trained model to predict semantic roles
    print("The 10 most frequency predicate-arg0 pairs: ", predict_with_huggingface('/Data/sub_amazon-sentences.txt'))