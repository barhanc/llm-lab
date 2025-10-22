import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, block_size: int, dim_model: int, num_heads: int, dropout_p: float):
        super().__init__()

        self.dim_head: int = dim_model // num_heads
        size = (num_heads, dim_model, self.dim_head)
        scale = 2 / (dim_model + self.dim_head) ** 0.5

        self.Wq = nn.Parameter(torch.normal(0.0, scale, size))
        self.Wk = nn.Parameter(torch.normal(0.0, scale, size))
        self.Wv = nn.Parameter(torch.normal(0.0, scale, size))

        self.proj = nn.Linear(dim_model, dim_model)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)) == 0)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.size()
        x = x.reshape(B, 1, T, -1)
        q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv

        scores: Tensor = (q @ k.transpose(-2, -1)) / self.dim_head**0.5
        scores = scores.masked_fill(self.tril[:T, :T], -torch.inf)  # type: ignore
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout1(scores)

        y = scores @ v
        y = y.transpose(2, 1).reshape(B, T, -1)
        y = self.proj(y)
        y = self.dropout2(y)

        return y


class Block(nn.Module):
    def __init__(self, block_size: int, dim_model: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.dim_head: int = dim_model // num_heads

        self.ln1 = nn.LayerNorm(dim_model)
        self.ln2 = nn.LayerNorm(dim_model)
        self.attn_layer = CausalMultiHeadSelfAttention(block_size, dim_model, num_heads, dropout_p)
        self.ffwd_layer = nn.Sequential(
            nn.Linear(dim_model, 4 * dim_model),
            nn.GELU(),
            nn.Linear(4 * dim_model, dim_model),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_layer(self.ln1(x))
        x = x + self.ffwd_layer(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        num_layers: int,
        dim_model: int,
        num_heads: int,
        dropout_p: float,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.block_size: int = block_size

        self.tok_emb = nn.Embedding(vocab_size, dim_model)
        self.pos_emb = nn.Embedding(block_size, dim_model)

        blocks = [Block(block_size, dim_model, num_heads, dropout_p) for _ in range(num_layers)]
        self.transformer = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.LayerNorm(dim_model), nn.Linear(dim_model, vocab_size))

        scale = (2 / dim_model) ** 0.5
        self.apply(lambda m: self._init_weights(m, scale))

    def _init_weights(self, m: nn.Module, scale: float):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=scale)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=scale)

    def forward(self, ctx: Tensor) -> Tensor:
        B, T = ctx.size()
        tok_emb = self.tok_emb(ctx)
        pos_emb = self.pos_emb(torch.arange(T).to(ctx))
        x = tok_emb + pos_emb
        x = self.transformer(x)
        x = self.head(x)
        return x

    @torch.no_grad()
    def generate(self, ctx: Tensor):
        self.eval()

        while True:
            ctx = ctx[-self.block_size :]
            logits = self(ctx.unsqueeze(0))
            logits = logits[0, -1, :]

            token = torch.multinomial(F.softmax(logits), num_samples=1)
            yield token.item()

            ctx = torch.cat([ctx, token])


if __name__ == "__main__":
    from tqdm import trange

    with open("data/mickiewicz.txt", encoding="utf-8") as file:
        text = file.read()
        print(text[:256])

    chars = sorted(set(text))
    vocab_size = len(chars)
    print(vocab_size, chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    data_train = data[: int(0.9 * len(data))]
    data_valid = data[int(0.9 * len(data)) :]

    def get_batch(data: Tensor, batch_size: int, block_size) -> tuple[Tensor, Tensor]:
        idx = torch.randint(len(data) - block_size, (batch_size,))
        src = torch.stack([data[i : i + block_size] for i in idx])
        tgt = torch.stack([data[i + 1 : i + 1 + block_size] for i in idx])
        return src, tgt

    @torch.no_grad()
    def eval(
        model: GPTLanguageModel,
        data: Tensor,
        num_evals: int,
        batch_size: int,
        device,
    ):
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

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device.upper()}")

    batch_size = 64
    block_size = 256
    log_period = 200
    num_epochs = 5_000
    num_evals = 200
    num_layers = 6
    num_heads = 6
    dim_model = 384
    dropout_p = 0.2

    loss_hist = {"train": {}, "valid": {}}

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        num_layers=num_layers,
        dim_model=dim_model,
        num_heads=num_heads,
        dropout_p=dropout_p,
    ).to(device)
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6} M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in (pbar := trange(num_epochs)):
        if (epoch + 1) % log_period == 0:
            loss = eval(model, data_valid, num_evals, batch_size, device)
            loss_hist["valid"][epoch] = loss
            print(f"epoch: {epoch+1:>5d}, valid loss: {loss}")

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "loss_hist": loss_hist,
                },
                f"checkpoints/chk_{epoch+1}.pt",
            )

            with open(f"checkpoints/out_{epoch+1}.txt") as f:
                f.write("".join(itos[t] for t in model.generate(torch.tensor([0]))))

        src, tgt = get_batch(data_train, batch_size, block_size)
        src, tgt = src.to(device), tgt.to(device)
        B, T = src.size()
        logits: Tensor = model(src)

        loss = F.cross_entropy(logits.reshape(B * T, -1), tgt.reshape(B * T))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist["train"][epoch] = loss.item()
        pbar.set_description(f"epoch: {epoch+1:>5d}, train loss: {loss.item():.4f}")
