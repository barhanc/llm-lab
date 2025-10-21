import torch
import torch.nn as nn
from torch.nn import functional as F


class KarpathyHead(nn.Module):
    """one head of self-attention (MODIFIED: NO CAUSAL MASK)"""

    def __init__(self, n_embd, head_size, dropout_rate):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Causal mask is NOT used for this test
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        # --- CAUSAL MASK REMOVED ---
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # Note: We won't test dropout on attention weights, so this won't be called in eval mode
        wei = self.dropout(wei)

        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class KarpathyMultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, n_embd, num_heads, dropout_rate):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([KarpathyHead(n_embd, head_size, dropout_rate) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


from nn import MultiHeadSelfAttention


def copy_weights(custom_model: MultiHeadSelfAttention, torch_model: KarpathyMultiHeadAttention):
    """
    Copies weights from your custom numpy-based model to the torch-based model.
    This is the most complex part, as it requires reshaping and transposing
    to match the different parameterization styles.
    """
    with torch.no_grad():
        # --- Copy Q, K, V weights ---
        # Your model: Wq is (num_heads, dim_model, dim_head)
        # Karpathy's: `num_heads` separate Linear layers, each with weight of shape (dim_head, dim_model)
        for i in range(torch_model.heads.__len__()):
            # Get the weights for the i-th head from your model
            w_q_head_i = custom_model.Wq[i]  # Shape: (dim_model, dim_head)
            w_k_head_i = custom_model.Wk[i]  # Shape: (dim_model, dim_head)
            w_v_head_i = custom_model.Wv[i]  # Shape: (dim_model, dim_head)

            # PyTorch nn.Linear weights are stored as (out_features, in_features).
            # To match the operation x @ W, we must transpose our (in, out) weights.
            torch_model.heads[i].query.weight.data = torch.from_numpy(w_q_head_i.T)
            torch_model.heads[i].key.weight.data = torch.from_numpy(w_k_head_i.T)
            torch_model.heads[i].value.weight.data = torch.from_numpy(w_v_head_i.T)

        # --- Copy final projection weights ---
        # Your model: map_out.layers[0].W is (dim_model, dim_model)
        # Karpathy's: proj.weight is (dim_model, dim_model)
        custom_proj_W = custom_model.map_out.layers[0].w
        custom_proj_b = custom_model.map_out.layers[0].b

        torch_model.proj.weight.data = torch.from_numpy(custom_proj_W.T)
        torch_model.proj.bias.data = torch.from_numpy(custom_proj_b)


import numpy as np
import time
import sys

if __name__ == "__main__":
    # --- Hyperparameters ---
    BATCH_SIZE = 4
    SEQ_LEN = 16
    DIM_MODEL = 16
    NUM_HEADS = 8
    DROPOUT_RATE = 0.1  # This will be disabled by .eval() and training=False
    DTYPE_NP = np.float32
    DTYPE_TORCH = torch.float32

    # --- 1. Initialize your custom model ---
    custom_model = MultiHeadSelfAttention(
        num_heads=NUM_HEADS,
        dim_model=DIM_MODEL,
        dropout=DROPOUT_RATE,
        dtype=DTYPE_NP,
    )

    # --- 2. Initialize Karpathy's model ---
    karpathy_model = KarpathyMultiHeadAttention(
        n_embd=DIM_MODEL,
        num_heads=NUM_HEADS,
        dropout_rate=DROPOUT_RATE,
    )

    print("Custom", sys.getsizeof(custom_model) , "B")
    print("Custom", sys.getsizeof(karpathy_model) , "B")

    # --- 3. Copy weights from your model to Karpathy's ---
    copy_weights(custom_model, karpathy_model)

    # --- 4. Set models to evaluation mode to disable dropout ---
    karpathy_model.eval()

    # --- 5. Create identical random inputs ---
    x_np = np.random.randn(BATCH_SIZE, SEQ_LEN, DIM_MODEL).astype(DTYPE_NP)
    x_torch = torch.from_numpy(x_np)

    # --- 6. Run forward pass on both models ---
    print("Running forward pass on both models...")
    t = time.perf_counter()
    y_custom = custom_model.forward(x_np, training=False)
    t = time.perf_counter() - t
    print("Custom", t)

    t = time.perf_counter()
    y_torch = karpathy_model.forward(x_torch)
    t = time.perf_counter() - t
    print("Torch", t)

    # --- 7. Compare the outputs ---
    y_torch_np = y_torch.detach().numpy()

    print(f"\nCustom model output shape: {y_custom.shape}")
    print(f"Torch model output shape:  {y_torch_np.shape}")

    # Use np.allclose for safe floating-point comparison
    are_equal = np.allclose(y_custom, y_torch_np, atol=1e-6)

    print(f"\nOutputs are identical:     {are_equal}")

    if not are_equal:
        max_diff = np.max(np.abs(y_custom - y_torch_np))
        print(f"Max absolute difference:   {max_diff:.8f}")
