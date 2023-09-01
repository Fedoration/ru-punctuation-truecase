from typing import List, Tuple
import pickle
import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import evaluate
metric = evaluate.load("seqeval")

import sys
sys.path.insert(0, 'D:\\progamming\\va\\truecase\\ru-punctuation-truecase\\src')
from process_text import clean_text, clean_text_3times

# ========== Data global variables ==========
PATH_TO_DATA = "D:\\progamming\\va\\truecase\\ru-punctuation-truecase\\data"

# ========== Model global variables ==========
MODEL_NAME = "DeepPavlov/rubert-base-cased-conversational"
# "DeepPavlov/rubert-base-cased-conversational" -> rubert-base-cased-conversational
SHORT_MODEL_NAME = MODEL_NAME.split('/')[1] if '/' in MODEL_NAME else MODEL_NAME
MODEL_MAX_LENGTH = 512


def file2array(path_to_file: str) -> List[List[str]]:
    result = []
    with open(path_to_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            result.append(line.split(' '))
    return result


def textlabel2arrays(path_to_text: str, path_to_labels: str) -> Tuple[List[List[str]], List[List[str]]]:
    texts = []
    labels = []
    with open(path_to_text, 'r', encoding='utf-8') as f_text:
        with open(path_to_labels, 'r', encoding='utf-8') as f_labels:
            for line_text, line_labels in zip(f_text.readlines(), f_labels.readlines()):

                line_text = line_text.strip()
                line_labels = line_labels.strip()
                
                texts.append(line_text.split(' '))
                labels.append(line_labels.split(' '))
    return texts, labels


def encode_tags(tags, tag2id, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def save_to_pickle(filename: str, data):
    # If file exists, delete it.
    if os.path.isfile(filename):
        os.remove(filename)
    else:
        print("Error: %s file not found" % filename)

    with open(f'{filename}', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename: str):
    data = None
    if os.path.isfile(filename):
        with open(f'{filename}', 'rb') as handle:
            data = pickle.load(handle)
    else:
        print("Error: %s file not found" % filename)
        
    return data


path_to_train_text = "../data/tatoeba_dataset/text_train.txt"
path_to_train_labels = "../data/tatoeba_dataset/labels_train.txt"
path_to_val_text = "../data/tatoeba_dataset/text_dev.txt"
path_to_val_labels = "../data/tatoeba_dataset/labels_dev.txt"

train_texts, train_tags = textlabel2arrays(path_to_train_text, path_to_train_labels)
val_texts, val_tags = textlabel2arrays(path_to_val_text, path_to_val_labels)

unique_labels = set(tag for doc in train_tags for tag in doc)
label_names = list(unique_labels)
label2id = {tag: id for id, tag in enumerate(label_names)}
id2label = {id: tag for tag, id in label2id.items()}

print(len(train_texts) == len(train_tags))
print(len(val_texts) == len(val_tags))
print(len(train_texts), len(train_tags))
print(len(val_texts), len(val_tags))

