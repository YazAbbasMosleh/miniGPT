# 🧠 Causal & Multi-Head Attention
### A Clear and Practical Tutorial — With Every Confusion Explained

> *Master shape tracking → Master attention.*

---

## 📌 Table of Contents

- [Big Picture](#-big-picture)
- [Input Tensor](#-input-tensor)
- [Linear Projections](#-linear-projections-q-k-v)
- [Splitting Into Multiple Heads](#-splitting-into-multiple-heads)
- [Confusion 1: Why transpose(1, 2)?](#-confusion-1-why-transpose12)
- [Confusion 2: Why keys.transpose(2, 3)?](#-confusion-2-why-keystranspose23)
- [Causal Attention — The Mask](#-causal-attention--the-mask)
- [Building the Mask](#-building-the-mask)
- [Applying the Mask](#-applying-the-mask)
- [Confusion 3: Why register_buffer?](#-confusion-3-why-register_buffer)
- [Computing the Context Vector](#-computing-the-context-vector)
- [Confusion 4: Why Transpose Again?](#-confusion-4-why-transpose-again)
- [Full Shape Flow](#-full-shape-flow)
- [Transpose Summary](#-transpose-summary)
- [Common Misconceptions](#-common-misconceptions)
- [Debug Tip](#-debug-tip)

---

## 🗺 Big Picture

In Transformer models, **attention** lets each token look at other tokens and decide which ones matter.

Two key mechanisms power this:

| Mechanism | What it does |
|---|---|
| **Causal Attention** | Prevents tokens from looking into the future |
| **Multi-Head Attention** | Learns multiple similarity metrics in parallel via representation subspaces |

---

## 📥 Input Tensor

Everything starts from:

```python
x.shape = (b, num_tokens, d_in)
```

| Symbol | Meaning |
|---|---|
| `b` | Batch size |
| `num_tokens` | Sequence length |
| `d_in` | Input embedding dimension |

---

## 🔢 Linear Projections (Q, K, V)

```python
keys    = self.W_key(x)    # → (b, num_tokens, d_out)
queries = self.W_query(x)  # → (b, num_tokens, d_out)
values  = self.W_value(x)  # → (b, num_tokens, d_out)

# where: d_out = num_heads × head_dim
```

Three independent linear layers project the input into keys, queries, and values.

---

## ✂️ Splitting Into Multiple Heads

We reshape using `.view()`:

```python
keys = keys.view(b, num_tokens, num_heads, head_dim)
# → (b, num_tokens, num_heads, head_dim)
```

> ⚠️ **Important:** `.view()` does **not** copy or split data.  
> It only reinterprets the memory layout.

---

## 🔀 Confusion 1: Why `transpose(1, 2)`?

```python
keys = keys.transpose(1, 2)
# (b, num_tokens, num_heads, head_dim)
# →
# (b, num_heads, num_tokens, head_dim)
```

**Why?** We want each head to operate independently.

After this transpose, the tensor hierarchy becomes:

```
Batch
  └── Head
        └── Token
              └── Feature
```

This makes per-head attention computation clean and parallelisable.

---

## ➗ Confusion 2: Why `keys.transpose(2, 3)`?

We compute attention scores like this:

```python
attn_scores = queries @ keys.transpose(2, 3)
```

Matrix multiplication requires inner dimensions to match:

```
(..., m, k) @ (..., k, n)
```

Before the transpose we have:

```
queries : (b, num_heads, num_tokens, head_dim)
keys    : (b, num_heads, num_tokens, head_dim)
                         ──────────  ─────────
                         ❌ these don't align for matmul
```

After `keys.transpose(2, 3)`:

```
keys    : (b, num_heads, head_dim, num_tokens)
                         ─────────  ──────────
queries : (b, num_heads, num_tokens, head_dim)
                                     ─────────  ✅ inner dims match

result  : (b, num_heads, num_tokens, num_tokens)
```

> `transpose(2, 3)` is purely a **linear algebra requirement** — nothing more.

---

## 🔭 Causal Attention — The Mask

In language modeling, when predicting token at position `i`, the model **must not** see tokens at positions `> i`. Otherwise it cheats.

```
Token:   0   1   2   3
         ↓   ↓   ↓   ↓
0  →  [ ✅  ❌  ❌  ❌ ]
1  →  [ ✅  ✅  ❌  ❌ ]
2  →  [ ✅  ✅  ✅  ❌ ]
3  →  [ ✅  ✅  ✅  ✅ ]
```

Each row = current token. Each column = token it can attend to.

---

## 🏗 Building the Mask

```python
self.register_buffer(
    "mask",
    torch.triu(torch.ones(context_length, context_length), diagonal=1)
)
```

**Step 1** — Create a matrix of ones:

```
1  1  1  1
1  1  1  1
1  1  1  1
1  1  1  1
```

**Step 2** — `torch.triu(..., diagonal=1)` keeps only the upper triangle (above the diagonal):

```
0  1  1  1
0  0  1  1
0  0  0  1
0  0  0  0
```

Where `1 = forbidden` and `0 = allowed`.

---

## 🙈 Applying the Mask

```python
attn_scores.masked_fill_(mask_bool, -torch.inf)
```

Wherever the mask is `1`, the attention score is replaced with `−∞`.

After softmax:

```
softmax(−∞) = 0
```

Future tokens receive **zero attention weight**. Perfect causal behaviour.

---

## 🗄 Confusion 3: Why `register_buffer`?

If you write:

```python
self.mask = tensor  # ❌ plain attribute
```

When you call `model.to("cuda")`, parameters move to GPU — but your mask stays on CPU, causing a **device mismatch error**.

`register_buffer` solves this:

```python
self.register_buffer("mask", tensor)  # ✅
```

| Property | Value |
|---|---|
| Moves with model (`.to(device)`) | ✅ Yes |
| Saved in `state_dict` | ✅ Yes |
| Learnable parameter | ❌ No |
| Has gradient | ❌ No |

> The mask is a **fixed architectural rule**, not a learned weight.

---

## 🧮 Computing the Context Vector

```python
# attn_weights : (b, num_heads, num_tokens, num_tokens)
# values       : (b, num_heads, num_tokens, head_dim)

context_vec = attn_weights @ values
# → (b, num_heads, num_tokens, head_dim)
```

Each head produces its own weighted combination of the values.

---

## 🔄 Confusion 4: Why Transpose Again?

After attention we have:

```
(b, num_heads, num_tokens, head_dim)
```

To merge the heads, we need `num_tokens` before `num_heads`:

```python
context_vec = context_vec.transpose(1, 2)
# (b, num_heads, num_tokens, head_dim)
# →
# (b, num_tokens, num_heads, head_dim)

context_vec = context_vec.view(b, num_tokens, d_out)
# → (b, num_tokens, d_out)
# where d_out = num_heads × head_dim
```

`.view()` here concatenates all head outputs into one flat vector per token.

---

## 📐 Full Shape Flow

```
Input         (b, num_tokens, d_in)
    │
    │  Linear (W_q, W_k, W_v)
    ▼
              (b, num_tokens, d_out)
    │
    │  .view()
    ▼
              (b, num_tokens, num_heads, head_dim)
    │
    │  .transpose(1, 2)
    ▼
              (b, num_heads, num_tokens, head_dim)
    │
    │  queries @ keys.transpose(2, 3)
    ▼
Attn scores   (b, num_heads, num_tokens, num_tokens)
    │
    │  masked_fill + softmax
    ▼
Attn weights  (b, num_heads, num_tokens, num_tokens)
    │
    │  @ values
    ▼
              (b, num_heads, num_tokens, head_dim)
    │
    │  .transpose(1, 2)
    ▼
              (b, num_tokens, num_heads, head_dim)
    │
    │  .view()
    ▼
Output        (b, num_tokens, d_out)
```

---

## 📊 Transpose Summary

There are exactly **two reasons** for transposing — and they are completely different:

| Transpose | Reason | Shape change |
|---|---|---|
| `transpose(1, 2)` | Rearrange for per-head operations & final reshape | `(b, T, H, D)` → `(b, H, T, D)` |
| `transpose(2, 3)` | Satisfy matrix multiplication inner-dimension rule | `(b, H, T, D)` → `(b, H, D, T)` |

> One is for **tensor layout**. The other is for **linear algebra**. Nothing mystical.

---

## ❌ Common Misconceptions

| Misconception | Reality |
|---|---|
| *"Transpose changes the data"* | No — it only reorders axes. Memory values are untouched. |
| *"`.view()` splits the tensor"* | No — it reinterprets the existing memory layout without copying. |
| *"The mask is learnable"* | No — it's a fixed rule. Registered as a buffer, no gradient. |
| *"Multi-head = multiple models"* | No — it splits one representation into parallel subspaces. |

---

## 🛠 Debug Tip

When attention code confuses you, follow these four steps:

```python
# 1. Print the current shape
print(tensor.shape)

# 2. Write the matmul rule: (..., m, k) @ (..., k, n)

# 3. Find which dimensions need to align

# 4. Ask: is this transpose for math compatibility
#         or for reshaping?
```

> **If you master shape tracking, you master attention.**

---

## 🧩 Deep Intuition

- **Causal Attention** = enforcing the direction of time
- **Multi-Head Attention** = learning multiple similarity metrics in parallel
- **Transposes** exist because linear algebra and tensor memory layout have different requirements

Once you track shapes carefully, the confusion disappears entirely.

---

*Made with 🖤 for anyone who has ever stared at a shape mismatch error at 2am.*