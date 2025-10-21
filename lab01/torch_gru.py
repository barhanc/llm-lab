import torch
import torch.nn as nn

from torch import Tensor


class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size: int, dim_model: int, num_layers: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.gru = nn.GRU(
            input_size=dim_model,
            hidden_size=dim_model,
            num_layers=num_layers,
            batch_first=True,
        )
        self.map_out = nn.Linear(dim_model, vocab_size)


class TransformerLanguageModel(nn.Module): ...
