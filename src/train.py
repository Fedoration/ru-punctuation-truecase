import os
import pickle
from typing import List, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import wandb
from config import (
    LEARNING_RATE,
    MODEL_MAX_LENGTH,
    MODEL_NAME,
    PATH_TO_DATA,
    SHORT_MODEL_NAME,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    VAL_BATCH_SIZE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from process_text import clean_text, clean_text_3times

LABEL_NAMES = []
METRIC = evaluate.load("seqeval")


def file2array(path_to_file: str) -> List[List[str]]:
    result = []
    with open(path_to_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            result.append(line.split(" "))
    return result


def textlabel2arrays(
    path_to_text: str, path_to_labels: str
) -> Tuple[List[List[str]], List[List[str]]]:
    texts = []
    labels = []
    with open(path_to_text, "r") as f_text:
        with open(path_to_labels, "r") as f_labels:
            for line_text, line_labels in zip(f_text.readlines(), f_labels.readlines()):
                line_text = line_text.strip()
                line_labels = line_labels.strip()

                texts.append(line_text.split(" "))
                labels.append(line_labels.split(" "))
    return texts, labels


def encode_tags(tags, tag2id, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def save_to_pickle(filename: str, data):
    # If file exists, delete it.
    if os.path.isfile(filename):
        os.remove(filename)
    else:
        print("Error: %s file not found" % filename)

    with open(f"{filename}", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename: str):
    data = None
    if os.path.isfile(filename):
        with open(f"{filename}", "rb") as handle:
            data = pickle.load(handle)
    else:
        print("Error: %s file not found" % filename)

    return data


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[LABEL_NAMES[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [LABEL_NAMES[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = METRIC.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


class CapitalizationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def main():
    # ========== Load dataset ==========
    print("Start loading dataset")

    path_to_train_text = f"{PATH_TO_DATA}/tatoeba_dataset/text_train.txt"
    path_to_train_labels = f"{PATH_TO_DATA}/tatoeba_dataset/labels_train.txt"
    path_to_val_text = f"{PATH_TO_DATA}/tatoeba_dataset/text_dev.txt"
    path_to_val_labels = f"{PATH_TO_DATA}/tatoeba_dataset/labels_dev.txt"

    train_texts, train_tags = textlabel2arrays(path_to_train_text, path_to_train_labels)
    val_texts, val_tags = textlabel2arrays(path_to_val_text, path_to_val_labels)

    unique_labels = set(tag for doc in train_tags for tag in doc)
    label_names = list(unique_labels)
    global LABEL_NAMES = label_names.copy()
    label2id = {tag: id for id, tag in enumerate(label_names)}
    id2label = {id: tag for tag, id in label2id.items()}

    assert len(train_texts) == len(
        train_tags
    ), "length of train texts not equal length of train tags"
    assert len(val_texts) == len(
        val_tags
    ), "length of val texts not equal length of val tags"

    print("Dataset has been loaded")

    # ========== Load tokens and correct labels ==========
    print("Start loading tokens")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, model_max_length=MODEL_MAX_LENGTH
    )

    # load train_encodings
    pickle_filename = f"{PATH_TO_DATA}/cached/train_encodings_{SHORT_MODEL_NAME}_{MODEL_MAX_LENGTH}.pickle"
    train_encodings = load_from_pickle(pickle_filename)
    if not train_encodings:
        # if there are no train encodings, calculate them
        train_encodings = tokenizer(
            train_texts,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
        )
        # save train_encodings
        save_to_pickle(pickle_filename, train_encodings)

    # load val_encodings
    pickle_filename = f"{PATH_TO_DATA}/cached/val_encodings_{SHORT_MODEL_NAME}_{MODEL_MAX_LENGTH}.pickle"
    val_encodings = load_from_pickle(pickle_filename)
    if not val_encodings:
        # if there are no val encodings, calculate them
        val_encodings = tokenizer(
            val_texts,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
        )
        # save val_encodings
        save_to_pickle(pickle_filename, val_encodings)

    # load train_labels
    pickle_filename = f"{PATH_TO_DATA}/cached/train_labels_{SHORT_MODEL_NAME}_{MODEL_MAX_LENGTH}.pickle"
    train_labels = load_from_pickle(pickle_filename)
    if not train_labels:
        # if there are no train labels, calculate them
        train_labels = encode_tags(train_tags, label2id, train_encodings)

        # save train labels
        save_to_pickle(pickle_filename, train_labels)

    # load val_labels
    pickle_filename = (
        f"{PATH_TO_DATA}/cached/val_labels_{SHORT_MODEL_NAME}_{MODEL_MAX_LENGTH}.pickle"
    )
    val_labels = load_from_pickle(pickle_filename)
    if not val_labels:
        # if there are no val labels, calculate them
        val_labels = encode_tags(val_tags, label2id, val_encodings)

        # save train labels
        save_to_pickle(pickle_filename, val_labels)

    print("Tokens has been loaded")

    # ========== Create dataset class ==========
    print("Start creating datset")

    train_encodings.pop("offset_mapping")
    val_encodings.pop("offset_mapping")
    train_dataset = CapitalizationDataset(train_encodings, train_labels)
    val_dataset = CapitalizationDataset(val_encodings, val_labels)

    print("Dataset has been created")

    # ========== Model ==========
    print("Start initializing model")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_names), id2label=id2label, label2id=label2id
    )

    for param in model.bert.parameters():
        param.requires_grad = False

    print("Model has been initialized")

    # ========== Train ==========
    print("Start training")

    wandb.init(
        project="huggingface-punctuation-and-capitalization",
        name=f"{SHORT_MODEL_NAME}-{MODEL_MAX_LENGTH}-{'tatoeba_dataset'}",
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    wandb.finish()
    print("Training has been finished")


if __name__ == "__main__":
    main()
