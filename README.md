![My Image](mini.png)
# 🤖 miniGPT — Building a GPT from Scratch



> A ground-up implementation of a GPT-style language model in PyTorch.  
> Every component — from tokenization to attention — written by hand, no black boxes.

![Status](https://img.shields.io/badge/status-in%20development-orange?style=flat-square)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![Framework](https://img.shields.io/badge/framework-PyTorch-red?style=flat-square)
![License](https://img.shields.io/badge/license-see%20LICENSE-green?style=flat-square)

---

## ⚠️ Work in Progress

This project is **actively under development**. The architecture is being built module by module — each folder represents a self-contained component of the full GPT pipeline. Not all pieces are connected yet, but every piece is being written from scratch with clarity and understanding as the goal.

---

## 🗺 Project Structure

```
miniGPT/
│
├── 📂 attention/              # Self-attention & multi-head attention
│   ├── self_attention.py      # Scaled dot-product self-attention
│   ├── multihead_attention.py # Multi-head + causal masking
│   ├── attention_readme.md    # Deep-dive tutorial on attention
│   └── attention_readme.html  # Rendered version
│
├── 📂 tokenization/           # Text → token IDs
│   ├── tokenizer_v1.py        # Simple regex-based tokenizer
│   ├── tokenizer_v2.py        # Improved tokenizer with special tokens
│   ├── bpe.py                 # Byte Pair Encoding implementation
│   └── bpe_test.py            # BPE unit tests
│
├── 📂 data/                   # Dataset & DataLoader utilities
│   ├── gpt_dataset_v1.py      # Sliding window token dataset
│   └── create_dataloader_v1.py# DataLoader factory with stride control
│
├── 📂 verdict/                # Raw text corpus (The Verdict)
│   ├── the-verdict.txt        # Training text
│   ├── download_verdict.py    # Script to fetch the corpus
│   ├── create_vocab_v1.py     # Vocabulary builder v1
│   ├── create_vocab_v2.py     # Vocabulary builder v2
│   ├── preprocessV1.py        # Text cleaning & preprocessing
│   └── verdict_test.py        # Sanity checks
│
├── README.md                  # ← You are here
├── requirements.txt           # Python dependencies
└── LICENSE
```

---

## 🧱 Architecture Overview

The full GPT pipeline being implemented, piece by piece:

```
Raw Text
    │
    ▼
┌─────────────┐
│ Tokenization│  → Convert text to integer token IDs
│  + Vocab    │  → Build vocabulary, handle special tokens, BPE
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Dataset &  │  → Sliding window sequences
│  DataLoader │  → Batch, shuffle, stride
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Embedding  │  → Token embeddings + positional embeddings
│   Layers    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Attention  │  → Causal self-attention
│   Module    │  → Multi-head attention  ✅ done
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Transformer │  → Attention + Feed-forward + LayerNorm  🔧 in progress
│   Block     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  GPT Model  │  → Stack of transformer blocks            🔧 in progress
│  (Full)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Training   │  → Loss, optimizer, training loop         📋 planned
│   Loop      │
└─────────────┘
```

---

## ✅ Progress Tracker

| Module | Component | Status |
|---|---|---|
| **Verdict Corpus** | Download & preprocess raw text | ✅ Done |
| **Tokenization** | Simple tokenizer v1 & v2 | ✅ Done |
| **Tokenization** | Byte Pair Encoding (BPE) | ✅ Done |
| **Data** | Sliding window GPT dataset | ✅ Done |
| **Data** | DataLoader with stride | ✅ Done |
| **Attention** | Scaled dot-product self-attention | ✅ Done |
| **Attention** | Causal masking | ✅ Done |
| **Attention** | Multi-head attention | ✅ Done |
| **Model** | Token + positional embeddings | ✅ Done |
| **Model** | Feed-forward layer | ✅ Done |
| **Model** | Transformer block (Attention + FF + Norm) | ✅ Done |
| **Model** | Full GPT model (stacked blocks) | ✅ Done |
| **Training** | Cross-entropy loss + optimizer | ✅ Done |
| **Training** | Training & validation loop | ✅ Done|
| **Inference** | Text generation / greedy decoding | 📋 Planned |
| **Inference** | Temperature sampling / top-k | 📋 Planned |

---

## 📂 Module Details

### 🔤 Tokenization — `tokenization/`

Converts raw text into sequences of integer token IDs.

- **`tokenizer_v1.py`** — Regex-based character-level tokenizer with a fixed vocabulary
- **`tokenizer_v2.py`** — Extended version with support for `<|unk|>` and `<|endoftext|>` special tokens
- **`bpe.py`** — Byte Pair Encoding: iteratively merges the most frequent character pairs to build a compact subword vocabulary

---

### 📦 Data Pipeline — `data/`

Prepares token sequences for training.

- **`gpt_dataset_v1.py`** — A `torch.utils.data.Dataset` that uses a sliding window over a token sequence to generate `(input, target)` pairs
- **`create_dataloader_v1.py`** — Wraps the dataset into a `DataLoader` with configurable batch size, stride, and shuffle

---

### 🧠 Attention — `attention/`

The core of the transformer. See [`attention/attention_readme.md`](attention/attention_readme.md) for a full deep-dive tutorial.

- **`self_attention.py`** — Scaled dot-product self-attention: Q, K, V projections, score computation, softmax
- **`multihead_attention.py`** — Full multi-head causal attention with:
  - Head splitting via `.view()` and `.transpose()`
  - Upper-triangular causal mask via `register_buffer`
  - Dropout on attention weights
  - Head merging and output projection

---

### 📖 Corpus — `verdict/`

Uses *The Verdict* as the training text — a short, clean English story ideal for small-scale language model experiments.

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/yourusername/miniGPT.git
cd miniGPT

# Install dependencies
pip install -r requirements.txt

# Download the corpus
python verdict/download_verdict.py

# Test the tokenizer
python tokenization/bpe_test.py

# Test the dataloader
python data/create_dataloader_v1.py
```

---

## 📚 Key Concepts Covered

- **Tokenization** — how raw text becomes numbers a model can process
- **Sliding window datasets** — how language models are trained on context windows
- **Self-attention** — how tokens communicate with each other
- **Causal masking** — how autoregressive models prevent future leakage
- **Multi-head attention** — how parallel representation subspaces improve expressiveness
- **`register_buffer`** — how to correctly handle non-learnable tensors in PyTorch modules

---

## 🤝 Contributing

This is a personal learning project, but suggestions, corrections, and discussions are very welcome. Open an issue or reach out directly.

---

## 📄 License

See [LICENSE](LICENSE) for details.

---

*Built from scratch, one tensor at a time. 🧱*WIP
