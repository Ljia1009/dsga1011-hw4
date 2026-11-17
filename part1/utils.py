import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

QWERTY_NEIGHBORS = {
    'a': 'qwsz', 'e': 'rdsw', 'i': 'uojk', 'o': 'iplk', 'u': 'yihj',
    's': 'awedx', 't': 'rfgy', 'n': 'bhjm', 'r': 'edft', 'l': 'kop'}


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


def _synonym(word):
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower() and ' ' not in name:
                syns.add(name)
    return random.choice(list(syns)) if syns else word


def _typo(word):
    if len(word) < 2:
        return word
    chars = list(word)
    idx = random.randrange(len(chars))
    c = chars[idx].lower()
    if c in QWERTY_NEIGHBORS:
        repl = random.choice(QWERTY_NEIGHBORS[c])
        chars[idx] = repl.upper() if chars[idx].isupper() else repl
    return ''.join(chars)

# Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


# Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    tokens = word_tokenize(example["text"])
    out = []
    for tok in tokens:
        r = random.random()
        if tok.isalpha() and r < 0.25:
            out.append(_synonym(tok))
        elif tok.isalpha() and r < 0.40:
            out.append(_typo(tok))
        else:
            out.append(tok)
    example["text"] = TreebankWordDetokenizer().detokenize(out)
    return example

    ##### YOUR CODE ENDS HERE ######
