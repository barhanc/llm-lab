import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score


class TransformerEncoderClassifier(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, n_heads, n_layers, n_classes, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, src: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        _, T = src.size()
        src = self.tok_emb(src) + self.pos_emb(torch.arange(T).to(src))
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        out = self.transformer(src, src_key_padding_mask=src_key_padding_mask)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).float()
            out = out * attention_mask
            out = out.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        else:
            out = torch.mean(out, dim=1)

        out = self.linear(out)
        return out

    @torch.no_grad()
    def predict_proba(self, src: Tensor) -> Tensor:
        self.eval()
        return torch.softmax(self(src), dim=-1)

    def predict(self, src: Tensor) -> Tensor:
        return torch.argmax(self.predict_proba(src), dim=-1)


if __name__ == "__main__":
    CONFIG = {
        "block_size": 128,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 3,
        "dropout": 0.3,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 100,
        "log_dir": "tce-slang/runs/custom_transformer_logs",
        "model_save_path": "tce-slang/best_custom_model.pth",
    }

    writer = SummaryWriter(log_dir=CONFIG["log_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {str(device).upper()}")

    print("Loading dataset...")
    DATASET = load_dataset("jziebura/polish_youth_slang_classification")
    DATASET = DATASET.rename_column("sentyment", "labels")

    def get_training_corpus(batch_size: int = 1000):
        dataset: Dataset = DATASET["train"]  # type: ignore
        for start_idx in range(0, len(dataset), batch_size):
            samples = dataset[start_idx : start_idx + batch_size]
            yield samples["tekst"]

    try:
        tokenizer = AutoTokenizer.from_pretrained("tce-slang/tokenizer")
        print("Tokenizer loaded.")
    except Exception:
        print("Training tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
        tokenizer = tokenizer.train_new_from_iterator(get_training_corpus(), vocab_size=1_000)
        tokenizer.save_pretrained("tce-slang/tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("tce-slang/tokenizer")

    def tokenize(examples):
        return tokenizer(
            examples["tekst"],
            padding="max_length",
            truncation=True,
            max_length=CONFIG["block_size"],
            return_tensors="pt",
        )

    print("Tokenizing dataset...")
    cols = ["tekst", "słowo slangowe", "znaczenie wyrazów slangowych", "źródło", "powiązana data"]
    dataset = DATASET.map(tokenize, batched=True)
    dataset = dataset.remove_columns(cols)
    dataset.set_format("torch")  # type: ignore

    train_loader = DataLoader(dataset["train"], batch_size=CONFIG["batch_size"], shuffle=True)  # type: ignore
    valid_loader = DataLoader(dataset["validation"], batch_size=CONFIG["batch_size"], shuffle=False)  # type: ignore

    model = TransformerEncoderClassifier(
        vocab_size=tokenizer.vocab_size,
        block_size=CONFIG["block_size"],
        d_model=CONFIG["d_model"],
        n_heads=CONFIG["n_heads"],
        n_layers=CONFIG["n_layers"],
        dropout=CONFIG["dropout"],
        n_classes=3,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()

    best_val_loss, global_step = float("inf"), 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss_accum, train_preds, train_targets = 0, [], []

        for batch in (pbar := tqdm(train_loader)):
            input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)

            logits = model(input_ids, mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_accum += loss.item()
            writer.add_scalar("Train/Loss_Step", loss.item(), global_step)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            global_step += 1

        model.eval()
        val_loss_accum, val_preds, val_targets = 0, [], []

        with torch.no_grad():
            for batch in valid_loader:
                input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
                mask = batch["attention_mask"].to(device)

                logits = model(input_ids, mask)
                loss = criterion(logits, labels)
                val_loss_accum += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        avg_train_loss = train_loss_accum / len(train_loader)
        avg_val_loss = val_loss_accum / len(valid_loader)

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="weighted")

        # --- TensorBoard Logging
        writer.add_scalar("Train/Loss_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        writer.add_scalar("Val/F1", val_f1, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"--> Best model saved to {CONFIG['model_save_path']}")

    writer.close()
