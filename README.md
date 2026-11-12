# LLM Lab

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Laboratory coursework, benchmarks, and experiments covering the Large Language Model lifecycle: causal language modeling, tokenization dynamics, parameter-efficient fine-tuning (PEFT/LoRA), memory-efficient training techniques, systematic LLM evaluation with diverse prompting strategies, and Tiny Reasoning Models (TRMs). Problem statements originate from [apohllo/computational-linguistics](https://github.com/apohllo/computational-linguistics).

## Overview

<details open>
<summary><b>Lab 1: Causal Language Modeling (RNN vs. Transformer)</b></summary>

- **Focus**: Causal language modeling and next-token prediction across recurrent and attention-based architectures.
- **Key Concepts**:
  - Gated Recurrent Unit (GRU) language model implementation and hidden-state handling.
  - Decoder-only Transformer (custom GPT) with masked causal multi-head self-attention.
  - Training dynamics, cross-entropy loss trajectories, and generation sampling on literary corpora (Adam Mickiewicz).
- **Directory**: [`lab01/`](./lab01/)
- **Notebook & Scripts**: [`lab01/notebook.ipynb`](./lab01/notebook.ipynb), [`lab01/gpt.py`](./lab01/gpt.py), [`lab01/gru.py`](./lab01/gru.py)
</details>

<details>
<summary><b>Lab 2: Tokenization Efficiency & Compression Benchmark</b></summary>

- **Focus**: Tokenization algorithms, text compression efficiency, and downstream language modeling impact.
- **Key Concepts**:
  - Comparative study of tokenization strategies: Character-level, Byte-Pair Encoding (BPE / Byte-level BPE), and pre-trained Hugging Face tokenizers.
  - Evaluation of fertility rate, compression ratio, vocabulary size trade-offs, and sequence length distributions.
  - Impact of tokenization granularity on causal language model training and perplexity.
- **Directory**: [`lab02/`](./lab02/)
- **Notebook & Scripts**: [`lab02/notebook.ipynb`](./lab02/notebook.ipynb), [`lab02/gpt.py`](./lab02/gpt.py)
</details>

<details>
<summary><b>Lab 3: Fine-Tuning vs. From-Scratch Training for Classification</b></summary>

- **Focus**: Decoder-only architectures for classification, pre-training transferability, and parameter-efficient tuning.
- **Key Concepts**:
  - Training a compact decoder-only Transformer from scratch with bounded capacity to limit overfitting.
  - Fine-tuning pre-trained transformer backbones using full parameter updates and parameter-efficient methods (PEFT / LoRA).
  - Empirical evaluation on Polish text classification benchmarks comparing accuracy, convergence speed, and parameter footprint.
- **Directory**: [`lab03/`](./lab03/)
- **Notebook & Scripts**: [`lab03/report.ipynb`](./lab03/report.ipynb), [`lab03/finetuning.py`](./lab03/finetuning.py), [`lab03/from_scratch.py`](./lab03/from_scratch.py)
</details>

<details>
<summary><b>Lab 4: Memory-Efficient Transformer Training Techniques</b></summary>

- **Focus**: Hardware memory optimization, throughput scaling, and execution profiling during Transformer training.
- **Key Concepts**:
  - Controlled benchmarking on WikiText-2 across activation checkpointing (gradient checkpointing), mixed precision (AMP fp16/bf16), FlashAttention-2, and torch compilation.
  - Profiling peak GPU VRAM allocation, step execution time, and maximum achievable batch size before Out-Of-Memory (OOM).
  - Throughput analysis (tokens/sec) and trade-offs between memory footprint and computational overhead.
- **Directory**: [`lab04/`](./lab04/)
- **Notebook & Scripts**: [`lab04/report.ipynb`](./lab04/report.ipynb), [`lab04/main.py`](./lab04/main.py)
</details>

<details>
<summary><b>Lab 5: LLM Evaluation across Diverse Prompting Strategies</b></summary>

- **Focus**: Systematic evaluation of open-source language models across diverse task families and prompting paradigms.
- **Key Concepts**:
  - Comparative benchmark between **Qwen 2.5 (1.5B)** (compact generalist) and **DeepSeek-R1 (7B)** (reasoning-oriented model).
  - Multi-task evaluation covering logical reasoning, mathematics, code generation, and factual extraction.
  - Prompt engineering comparison: Zero-shot, Few-shot in-context learning, and Chain-of-Thought (CoT) reasoning prompts.
- **Directory**: [`lab05/`](./lab05/)
- **Notebook & Scripts**: [`lab05/report.ipynb`](./lab05/report.ipynb), [`lab05/main.py`](./lab05/main.py), [`lab05/results.json`](./lab05/results.json)
</details>

<details>
<summary><b>Lab 6: Tiny Reasoning Models (TRM) & Constraint Satisfaction</b></summary>

- **Focus**: Compact recursive reasoning architectures for high-intensity logical constraint satisfaction.
- **Key Concepts**:
  - Evaluation of Tiny Reasoning Models (TRM) on complex discrete constraint tasks (4x4 Sudoku).
  - Comparing recursive latent reasoning against standard autoregressive large language models.
  - Exact match accuracy, reasoning trace efficiency, and compute/parameter trade-offs in resource-constrained environments.
- **Directory**: [`lab06/`](./lab06/)
- **Notebook**: [`lab06/notebook.ipynb`](./lab06/notebook.ipynb)
</details>

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/barhanc/llm-lab.git
cd llm-lab
```

### 2. Environment Setup

Manage dependencies and runtime environments using [uv](https://docs.astral.sh/uv/):

```bash
# Synchronize root environment and install dependencies from lockfile
uv sync
```

Alternatively, individual labs (`lab04`, `lab05`, `lab06`) contain dedicated `pyproject.toml` configurations that can be run directly:

```bash
cd lab04
uv sync
```

### 3. Launching Notebooks

Launch interactive notebooks or execute standalone lab scripts:

```bash
# Run Jupyter Lab
uv run jupyter lab

# Execute specific lab scripts
uv run python lab01/gpt.py
uv run python lab04/main.py
```

## References

- [Computational Linguistics Course Repository (apohllo)](https://github.com/apohllo/computational-linguistics) — Original course problem statements and assignments.
