from typing import Dict, Optional

import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

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


class BertTokenClassification(nn.Module):
    def __init__(
        self,
        pretrained_model: str,
        targets: Dict[str, int],
        freeze_pretrained: Optional[bool] = False,
        lstm_dim: int = -1,
        *args,
        **kwargs
    ) -> None:
        super(BertTokenClassification, self).__init__()
        self.pretrained_transformer = AutoModel.from_pretrained(MODEL_NAME)

        if freeze_pretrained:
            for p in self.pretrained_transformer.parameters():
                p.requires_grad = False

        bert_dim = 768

        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=bert_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
        )

        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=len(targets))
