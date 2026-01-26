#!/bin/bash
set -e

echo "Step: Cloning and Setup..."
if [ ! -d "trm_sudoku" ]; then
    git clone https://github.com/allthingssecurity/trm_sudoku.git
fi
cd trm_sudoku
uv sync

echo "Step: Training TRM..."
uv run python src/nn/train.py \
  experiment=trm_sudoku4x4 \
  trainer=gpu trainer.accelerator=cuda trainer.precision=32-true \
  timekeeping.max_epochs=60 timekeeping.batch_size=128 \
  model_tuning.hidden_size=128 model_tuning.num_layers=2 \
  model_tuning.N_supervision=2 \
  model_tuning.learning_rate=3e-4 model_tuning.learning_rate_emb=3e-3

LATEST_RUN=$(ls -td train/runs/*/ | head -1)
TIMESTAMP=$(basename "$LATEST_RUN")
export CHECKPOINT_PATH="${LATEST_RUN}checkpoints/last.ckpt"
export METRICS_CSV="${LATEST_RUN}csv/version_0/metrics.csv"

echo "Detected latest run: $TIMESTAMP"

echo "Step: Running Validation..."
uv run python - <<PY
import os
import torch
from lightning import Trainer
from src.nn.data.sudoku4x4_datamodule import SudokuDataModule
from src.nn.models.trm_module import TRMModule

ckpt = os.getenv('CHECKPOINT_PATH')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dm = SudokuDataModule(
    batch_size=128, num_workers=0,
    grid_size=4, max_grid_size=6,
    generate_on_fly=True,
    num_train_puzzles=2000, num_val_puzzles=800,
)
dm.setup('fit')

model = TRMModule.load_from_checkpoint(ckpt, map_location=device)
trainer = Trainer(accelerator=device, devices=1)
results = trainer.validate(model, dm.val_dataloader())
print("\nFinal Validation Results:", results)
PY

echo "Step: Visualizing Inference Samples..."
uv run python - <<PY
import torch, numpy as np, os
from src.nn.data.sudoku4x4_datamodule import SudokuDataModule
from src.nn.models.trm_module import TRMModule

def decode_token(t):
    if t <= 2: return 0
    return int(t-2)

ckpt = os.getenv('CHECKPOINT_PATH')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TRMModule.load_from_checkpoint(ckpt, map_location=device).to(device).eval()

N=5; GRID=4; MAX=6
dm = SudokuDataModule(batch_size=N, num_workers=0, grid_size=4, max_grid_size=6,
                      generate_on_fly=True, num_train_puzzles=2000, num_val_puzzles=N)
dm.setup('fit')
batch = next(iter(dm.val_dataloader()))

for k,v in batch.items():
    if isinstance(v, torch.Tensor): batch[k] = v.to(device)

with torch.no_grad():
    carry = model.initial_carry(batch)
    while True:
        carry, outputs = model.forward(carry, batch)
        if carry.halted.all(): break
    preds = outputs['logits'].argmax(dim=-1)

labels = batch['output']
for i in range(N):
    pred4 = np.vectorize(decode_token)(preds[i].cpu().numpy().reshape(MAX,MAX))[:GRID,:GRID]
    lab4  = np.vectorize(decode_token)(labels[i].cpu().numpy().reshape(MAX,MAX))[:GRID,:GRID]
    inp4  = np.vectorize(decode_token)(batch['input'][i].cpu().numpy().reshape(MAX,MAX))[:GRID,:GRID]
    print(f'Sample {i+1} (Match: {np.array_equal(pred4, lab4)})')
    print(f'Input:\n{inp4}\nPred:\n{pred4}\n' + '-'*20)
PY