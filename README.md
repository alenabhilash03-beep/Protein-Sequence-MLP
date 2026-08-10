# Human Protein Sequence Language Model 

A character-level Multi-Layer Perceptron (MLP) language model implemented in PyTorch that auto-regressively generates human protein sequences. Based on the architecture by Bengio et al. (2003), this model uses a context window of 10 amino acids to predict the next amino acid in a sequence using data fetched directly from UniProtKB.

---

## Features

* **UniProtKB Integration:** Automated fetching and parsing of reviewed human proteomic data.
* **Character-Level Tokenization:** Vocabulary mapping for all standard/non-standard amino acids using '.' as a sequence start/stop token.
* **Fixed Context Window:** Sliding window of 10 amino acids to construct pairs of vectorized context and targets.
* **Custom Initialization:** He initialization for hidden weights and scaled output projections to optimize initial cross-entropy loss.
* **Auto-Regressive Sampling:** Generates synthetic protein sequences character-by-character.

---

## Installation

### Prerequisites

* Python 3.8+

### Dependencies

Install the required Python libraries using pip:

```bash
pip install torch numpy matplotlib

```

---

## Quickstart

```python
import torch
from protein_mlp import train_mlp, generate_protein

# 1. Train the character-level MLP model on protein sequences
parameters, stoi, itos = train_mlp(
    data_path="human_proteins.txt",
    block_size=10,
    emb_dim=16,
    hidden_dim=300,
    iterations=50000,
    batch_size=128
)

# 2. Auto-regressively generate a synthetic protein sequence
sequence = generate_protein(parameters, stoi, itos, temperature=1.0)
print(f"Generated Sequence:\n{sequence}")

```

---

## Auto-Regressive Sampling

Protein sequence generation is performed character-by-character in an auto-regressive feedback loop:

1. **Context Initialization:** Initialize a sliding context window of length k = 10 using padding tokens ('.').
2. **Logits Calculation:** Compute output logits for the current context (forward pass)
3. **Multinomial Sampling:** Apply Softmax to  logits to derive a probability distribution P, then sample the next amino acid index.
4. **Window Shift & Termination:** Append the sampled character to the sequence, drop the oldest token from the context, shift the window left, and repeat until the delimiter ('.') is sampled.

---

## Architecture & Hyperparameters

| Parameter | Value / Setting | Description |
| --- | --- | --- |
| **Context Window ($k$)** | 10 | Number of prior amino acids used as prediction context |
| **Embedding Dimension** | 16 | Vector size assigned to each character in the vocabulary |
| **Hidden Layer** | 300 units ($\tanh$) | Maps flattened lookup embeddings ($10 \times 16 = 160$) |
| **Optimization** | Minibatch SGD | Learning rate $\eta = 0.1$, batch size = $128$, $100,000$ iterations, then learning rate $\eta = .01$, batch size = $128$, $50,000$ iterations|
| **Data Split Ratio** | 80% / 10% / 10% | Train, dev, and test set partitioning (~9.1M sample pairs) |
