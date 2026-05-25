import re
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

PAD_IDX = 0


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout=0.3,
        pretrained_embeddings=None,
        freeze_embeddings=False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        # assign glove weights to embedding layer
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)

        # optionally freeze embedding weights to prevent updates during training
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # bidirectional = double hidden dim
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, lengths):
        # input_ids: (batch, seq_len)
        x = self.dropout(self.embedding(input_ids))  # (batch, seq_len, embed_dim)

        # packing makes LSTM efficiently skips PAD tokens
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        output, (hidden, _) = self.lstm(packed)  # hidden: (num_layers*2, batch, hidden)

        # concat last forward and backward hidden states
        # hidden[-2] = last forward layer, hidden[-1] = last backward layer
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden*2)
        last_hidden = self.dropout(last_hidden)

        return self.classifier(last_hidden)  # (batch, num_classes)
