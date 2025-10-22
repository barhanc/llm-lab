import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, block_size: int, dim_model: int, num_heads: int, dropout_p: float):
        super().__init__()
        assert dim_model % num_heads == 0

        self.dim_model: int = dim_model
        self.num_heads: int = num_heads
        self.dropout_p: float = dropout_p

        self.l_attn = nn.Linear(dim_model, 3 * dim_model, bias=False)
        self.l_proj = nn.Linear(dim_model, dim_model)
        self.d_attn = nn.Dropout(dropout_p)
        self.d_proj = nn.Dropout(dropout_p)

        self.flash: bool = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            )

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        q, k, v = self.l_attn(x).split(self.dim_model, dim=2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        if self.flash:
            dropout_p = self.dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
        else:
            dh = C // self.num_heads
            scores = (q @ k.transpose(-2, -1)) / dh**0.5
            scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, -torch.inf)  # type:ignore
            scores = F.softmax(scores, dim=-1)
            scores = self.d_attn(scores)
            out = scores @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.d_proj(self.l_proj(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, dim_model: int, dropout_p: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_model, 4 * dim_model),
            nn.GELU(),
            nn.Linear(4 * dim_model, dim_model),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, block_size: int, dim_model: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim_model)
        self.attn = MultiHeadSelfAttention(block_size, dim_model, num_heads, dropout_p)
        self.ln_2 = nn.LayerNorm(dim_model)
        self.ffwd = FeedForward(dim_model, dropout_p)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
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
        self.transformer = nn.Sequential(
            *[Block(block_size, dim_model, num_heads, dropout_p) for _ in range(num_layers)]
        )

        self.layer_norm = nn.LayerNorm(dim_model)
        self.linear = nn.Linear(dim_model, vocab_size)

        self.apply(lambda m: self._init_weights(m))

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ctx: Tensor) -> Tensor:
        _, T = ctx.size()

        tok_emb = self.tok_emb(ctx)
        pos_emb = self.pos_emb(torch.arange(T).to(ctx))

        out = tok_emb + pos_emb
        out = self.transformer(out)
        out = self.linear(self.layer_norm(out))

        return out

    @torch.no_grad()
    def generate(self, ctx: Tensor, out_size: int | None = None):
        self.eval()
        i = 0
        while True:
            ctx = ctx[-self.block_size :]
            logits = self(ctx.unsqueeze(0))
            logits = logits[0, -1, :]

            token = torch.multinomial(F.softmax(logits, dim=0), num_samples=1)
            yield token.item()

            ctx = torch.cat([ctx, token])

            i += 1
            if out_size and i == out_size:
                break


if __name__ == "__main__":
    from tqdm import trange

    with open("data/mickiewicz.txt", encoding="utf-8") as file:
        text = file.read()

    chars = sorted(set(text))
    vocab_size = len(chars)

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
    num_layers = 6
    num_evals = 200
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
        if epoch % log_period == 0 or epoch == num_epochs - 1:
            loss = eval(model, data_valid, num_evals, batch_size, device)
            loss_hist["valid"][epoch] = loss
            print(f"epoch: {epoch+1:>5d}, valid loss: {loss:.4f}")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "loss_hist": loss_hist,
                },
                f"checkpoints/chk_{epoch}.pt",
            )

            with open(f"checkpoints/out_{epoch}.txt", "w", encoding="utf-8") as f:
                f.write("".join(itos[t] for t in model.generate(torch.tensor([0]).to(device), 1_000)))

        model.train()
        src, tgt = get_batch(data_train, batch_size, block_size)
        src, tgt = src.to(device), tgt.to(device)
        B, T = src.size()
        logits: Tensor = model(src)

        loss = F.cross_entropy(logits.reshape(B * T, -1), tgt.reshape(B * T))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist["train"][epoch] = loss.item()
        pbar.set_description(f"epoch: {epoch:>5d}, train loss: {loss.item():.4f}")
