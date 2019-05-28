import os

from pytorch_pretrained_bert.modeling import BertModel
from torch import nn


class BertModule(nn.Module):
    def __init__(self, bert_model_name, dropout_prob=0.1, cache_dir="./cache/"):
        super().__init__()

        self.dropout = nn.Dropout(dropout_prob)

        # Create cache directory if not exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.bert_model = BertModel.from_pretrained(
            bert_model_name, cache_dir=cache_dir
        )

    def forward(self, token_ids, token_segments=None):
        encoded_layers, pooled_output = self.bert_model(token_ids, token_segments)
        pooled_output = self.dropout(pooled_output)
        return encoded_layers, pooled_output
