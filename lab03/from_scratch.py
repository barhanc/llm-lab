import torch
import torch.nn as nn
from torch import Tensor


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, n_heads, n_layers, n_classes):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, src: Tensor) -> Tensor:
        _, T = src.size()
        src = self.tok_emb(src) + self.pos_emb(torch.arange(T).to(src))
        out = self.transformer(src)
        out = torch.mean(out, dim=1)
        out = self.linear(out)
        return out

    @torch.no_grad()
    def predict_proba(self, src: Tensor) -> Tensor:
        self.eval()
        return torch.softmax(self(src), dim=-1)

    def predict(self, src: Tensor) -> Tensor:
        return torch.argmax(self.predict_proba(src), dim=-1)


def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]
