import re
import string
from typing import List, Tuple

import torch
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer


class TextHandler:
    """A class that performs some manipulations with text, such as splitting text into words, converting to lowercase, removing punctuation marks, etc."""

    @staticmethod
    def split_texts_into_words(texts: List[str]) -> List[List[str]]:
        """Splits each text in the batch into words separated by spaces

        Args:
            texts (List[str]): array of texts

        Returns:
            List[List[str]]: returns an array of texts divided into words
        """
        texts_words = []
        for text in texts:
            words = text.split(" ")
            words = [word for word in words if word]
            texts_words.append(words)
        return texts_words

    @staticmethod
    def remove_punctuation(line: str) -> str:
        """Removes punctuation from the input text

        Args:
            line (str): input text

        Returns:
            str: input text without punctuation
        """
        all_punct_marks = string.punctuation + "«»—"

        line = (
            line.replace("...", ".")
            .replace("…", ".")
            .replace("—", "—")
            .replace("―", "—")
            .replace("?!", "?")
            .replace("!?", "?")
        )
        line = re.sub("[-‐–]", "-", line)

        for c in all_punct_marks:
            line = line.replace(c, "")

        line = re.sub("[ \t]+", " ", line)
        line = re.sub("[" + all_punct_marks + "]", "", line)
        return line.strip()

    @staticmethod
    def convert_to_lowercase_and_remove_punctuation(line: str) -> str:
        """Converts the input text to lowercase and removes punctuation

        Args:
            line (str): input text

        Returns:
            str: input text in lowercase without punctuation
        """
        line = line.lower()
        line = TextHandler.remove_punctuation(line)
        return line

    @staticmethod
    def convert_to_lowercase_and_remove_punctuation_on_batch(
        lines: List[str],
    ) -> List[str]:
        """Converts a batch of texts to lowercase and removes punctuation marks

        Args:
            lines (List[str]): a batch of texts

        Returns:
            List[str]: a batch of texts in lowercase without punctuation
        """
        return [
            TextHandler.convert_to_lowercase_and_remove_punctuation(line)
            for line in lines
        ]


class ReCapitalizationModel:
    def __init__(
        self, path_to_checkpoint, model_name, model_max_length, is_question=False
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, model_max_length=model_max_length
        )
        self.token_classifier = AutoModelForTokenClassification.from_pretrained(
            path_to_checkpoint
        )
        self.is_question = is_question

    def _tokenize_texts(self, texts_words: List[List[str]]):
        inputs = self.tokenizer(
            texts_words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def _predict_token_classification(self, inputs):
        with torch.no_grad():
            logits = self.token_classifier(**inputs).logits

        predictions = torch.argmax(logits, dim=2)
        return predictions

    def restore_capitalization(self, texts: List[str]) -> List[str]:
        texts = TextHandler.convert_to_lowercase_and_remove_punctuation_on_batch(texts)
        texts_words = TextHandler.split_texts_into_words(texts)

        inputs = self._tokenize_texts(texts_words)
        predictions = self._predict_token_classification(inputs)

        truecase_texts = []
        for text, model_input, predict in zip(
            texts_words, inputs.encodings, predictions
        ):
            predicted_token_class = [
                self.token_classifier.config.id2label[t.item()] for t in predict
            ]

            word_class = {}
            for word_id, token_class in zip(
                model_input.word_ids, predicted_token_class
            ):
                if (word_id is not None) and (word_id not in word_class):
                    word_class[word_id] = token_class

            truecase_words = []
            for i, word in enumerate(text):
                try:
                    is_upper = word_class[i] == "U"
                except Exception as e:
                    is_upper = False
                if is_upper:
                    truecase_word = word.capitalize()
                else:
                    truecase_word = word

                truecase_words.append(truecase_word)

            # Если требуется составить вопрос
            if self.is_question:
                truecase_texts.append(" ".join(truecase_words) + "?")
            else:
                truecase_texts.append(" ".join(truecase_words))
        return truecase_texts
