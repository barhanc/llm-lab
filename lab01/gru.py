import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Generator, Any

type DeviceLikeType = str | torch.device


class GRULanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        num_layers: int,
        dim_model: int,
        dropout_p: float,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.block_size: int = block_size

        self.tok_emb = nn.Embedding(vocab_size, dim_model)
        self.out_map = nn.Linear(dim_model, vocab_size)
        self.rnn = nn.GRU(
            dim_model,
            dim_model,
            num_layers,
            dropout=dropout_p,
            batch_first=True,
        )

    def forward(self, ctx: Tensor) -> Tensor:
        tok_emb = self.tok_emb(ctx)
        out, _ = self.rnn(tok_emb)
        out = self.out_map(out)
        return out

    @torch.no_grad()
    def generate(self, ctx: Tensor, out_size: Optional[int] = None) -> Generator[int, Any, None]:
        self.eval()
        i = 0
        while True:
            ctx = ctx[-self.block_size :]
            logits = self(ctx.unsqueeze(0))
            logits = logits[0, -1, :]
            token = torch.multinomial(F.softmax(logits, dim=0), num_samples=1)
            yield token.item()  # type: ignore

            ctx = torch.cat([ctx, token])

            i += 1
            if out_size and i == out_size:
                break


def get_batch(data: Tensor, batch_size: int, block_size: int) -> tuple[Tensor, Tensor]:
    idx = torch.randint(len(data) - block_size, (batch_size,))
    src = torch.stack([data[i : i + block_size] for i in idx])
    tgt = torch.stack([data[i + 1 : i + 1 + block_size] for i in idx])
    return src, tgt
