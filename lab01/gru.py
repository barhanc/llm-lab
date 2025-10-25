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


@torch.no_grad()
def eval(model: GRULanguageModel, data: Tensor, num_evals: int, batch_size: int, device: DeviceLikeType) -> float:
    model.eval()
    running_loss = 0.0
    for _ in range(num_evals):
        src, tgt = get_batch(data, batch_size, model.block_size)
        src, tgt = src.to(device), tgt.to(device)

        B, T = src.size()
        logits: Tensor = model(src)
        loss = F.cross_entropy(logits.reshape(B * T, -1), tgt.reshape(B * T))
        running_loss += loss.item()

    return running_loss / num_evals


if __name__ == "__main__":
    from tqdm import trange

    with open("data/mickiewicz.txt", encoding="utf-8") as file:
        text = file.read()

    chars = sorted(set(text))
    vocab_size = len(chars)

    itos = {i: ch for i, ch in enumerate(chars)}
    stoi = {ch: i for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    data_train = data[: int(0.9 * len(data))]
    data_valid = data[int(0.9 * len(data)) :]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    batch_size = 64
    block_size = 256
    log_period = 500
    num_epochs = 5000
    num_layers = 6
    num_evals = 200
    dim_model = 512
    dropout_p = 0.2

    loss_hist = {"train": {}, "valid": {}}

    model = GRULanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        num_layers=num_layers,
        dim_model=dim_model,
        dropout_p=dropout_p,
    ).to(device)
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in (pbar := trange(num_epochs)):
        model.train()
        src, tgt = get_batch(data_train, batch_size, block_size)
        src, tgt = src.to(device), tgt.to(device)
        B, T = src.size()
        logits: Tensor = model(src)

        loss = F.cross_entropy(logits.reshape(B * T, -1), tgt.reshape(B * T))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # --- Logging and evaluation ---

        loss_hist["train"][epoch] = loss.item()
        pbar.set_description(f"epoch: {epoch:>5d}, train loss: {loss.item():.4f}")

        if epoch % log_period == 0 or epoch == num_epochs - 1:
            loss = eval(model, data_valid, num_evals, batch_size, device)
            loss_hist["valid"][epoch] = loss
            print(f"epoch: {epoch:>5d}, valid loss: {loss:.4f}")

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "loss": loss_hist,
                },
                f"checkpoints/gru/chk_{epoch}.pt",
            )

            with open(f"checkpoints/gru/out_{epoch}.txt", "w", encoding="utf-8") as f:
                ctx = torch.tensor([0], dtype=torch.long).to(device)
                out_size = 512
                f.write("".join(itos[tok] for tok in model.generate(ctx, out_size)))
