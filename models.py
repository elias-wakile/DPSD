import torch
from torch import nn
from opacus.layers import DPLSTM
from opacus import layers


class AttentionClassifier(torch.nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_of_heads,
                 dropout, seq_length, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.hidden_size = hidden_size
        self.lstm = layers.DPLSTM(embedding_size, hidden_size,
                                  batch_first=True, dropout=dropout,
                                  bidirectional=True, num_layers=num_layers)
        self.attention = layers.DPMultiheadAttention(hidden_size,
                                                     num_of_heads)
        self.flatten = nn.Flatten(start_dim=1)
        self.output = nn.Linear(seq_length * hidden_size, 1)

    def forward(self, inp, hidden=None):
        x = self.embedding(inp)
        vec, _ = self.lstm(x, hidden)
        x = vec[:, :, :self.hidden_size] + vec[:, :, self.hidden_size:]
        x, weights = self.attention(x, x, x)
        x = self.flatten(x)
        x = self.output(x)
        return x


class LSTMModel(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_size,
                 hidden_layer, num_layers,
                 output_layer=1, is_bidirectional=True, dropout=0.1):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_layer
        self.output_layer = output_layer
        self.lstm_layers = num_layers
        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.lstm_model = DPLSTM(embedding_size, hidden_layer,
                                 num_layers=num_layers,
                                 bidirectional=is_bidirectional,
                                 batch_first=True, dropout=dropout)
        self.output_layer = torch.nn.Linear(2 * hidden_layer, output_layer)

    def forward(self, inp, hidden=None):
        embedded = self.embedding(inp)
        vec, _ = self.lstm_model(embedded, hidden)
        nor_out = vec[:, -1, :self.hidden_size]
        rev_out = vec[:, 0, self.hidden_size:]
        vec = torch.concat([nor_out, rev_out], dim=1)
        vec = self.output_layer(vec)
        vec = torch.squeeze(vec, dim=-1)
        return vec
