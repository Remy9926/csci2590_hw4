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


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
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

    keyboard_nearest = {
    'a': ['a', 'q', 'w', 's', 'z'],
    'b': ['b', 'v', 'g', 'h', 'n'],
    'c': ['c', 'x', 'd', 'f', 'v'],
    'd': ['d', 's', 'e', 'r', 'f', 'v', 'c', 'x'],
    'e': ['e', 'w', 's', 'd', 'r'],
    'f': ['f', 'd', 'r', 't', 'g', 'b', 'v', 'c'],
    'g': ['g', 'f', 't', 'y', 'h', 'n', 'b', 'v'],
    'h': ['h', 'g', 'y', 'u', 'j', 'm', 'n', 'b'],
    'i': ['i', 'u', 'j', 'k', 'o'],
    'j': ['j', 'h', 'u', 'i', 'k', 'l', 'm', 'n'],
    'k': ['k', 'j', 'i', 'o', 'l', 'm'],
    'l': ['l', 'k', 'o', 'p'],
    'm': ['m', 'n', 'j', 'k'],
    'n': ['n', 'b', 'h', 'j', 'm'],
    'o': ['o', 'i', 'k', 'l', 'p'],
    'p': ['p', 'o', 'l'],
    'q': ['q', 'w', 'a'],
    'r': ['r', 'e', 'd', 'f', 't'],
    's': ['s', 'a', 'w', 'e', 'd', 'c', 'x', 'z'],
    't': ['t', 'r', 'f', 'g', 'y'],
    'u': ['u', 'y', 'h', 'j', 'i'],
    'v': ['v', 'c', 'f', 'g', 'b'],
    'w': ['w', 'q', 'a', 's', 'e'],
    'x': ['x', 'z', 's', 'd', 'c'],
    'y': ['y', 't', 'g', 'h', 'u'],
    'z': ['z', 'a', 's', 'x'],
    }
    # remove breaks = 0.1
    # synonyms = 0.25
    # shuffle sentences - 0.3
    # typos = 0.35

    text = example["text"]
    get_prob = random.random()

    if get_prob <= 0.1:
        # remove_breaks
        text = text.split("<br />")
        new_text = "".join(text).strip()
    elif get_prob <= 0.35:
        # synonym
        replacement_prob = random.random()
        split_text = text.split(" ")

        for i in range(len(split_text)):
            drawn_prob = random.random()
            if drawn_prob <= replacement_prob:
                word = split_text[i]
                lemmas = wordnet.synsets(word)

                if lemmas != []:
                    index = random.randint(0, len(lemmas) - 1)
                    lemma_names = lemmas[index].lemma_names()
                    index = random.randint(0, len(lemma_names) - 1)
                    new_word = lemma_names[index]
                    split_text[i] = new_word
        new_text = " ".join(split_text)
    elif get_prob <= 0.65:
        # random sentence permutation
        split_break = text.split("<br />")
        text = " ".join(split_break)
        words = text.split(".")
        random.shuffle(words)
        new_text = ".".join(words).strip()
    else:
        # typo
        text = text.split("<br />")
        text = " ".join(text).strip()
        typos = {}
        typo_prob = 0.3

        for i in range(len(text)):
            draw_prob = random.random()
            if draw_prob <= typo_prob and (i != 0 and text[i - 1] != " ") and (i < len(text) - 1 and text[i + 1] not in " .?!"):
                original_char = text[i]
                
                if original_char.lower() in keyboard_nearest:
                    nearest = keyboard_nearest[original_char.lower()].copy()
                    random.shuffle(nearest)
                    new_char = nearest[0]
                    if original_char != original_char.lower():
                        new_char = new_char.upper()
                    typos[i] = new_char
        
        new_string = []
        for i in range(len(text)):
            if i in typos:
                new_string.append(typos[i])
            else:
                new_string.append(text[i])

        new_text = "".join(new_string)

    example["text"] = new_text

    ##### YOUR CODE ENDS HERE ######

    return example
