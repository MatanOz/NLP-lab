# === Homework 3: Self-Attention and Multi-Head Attention ===

# Step 1: Import Required Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# Step 2: Implement Self-Attention
def self_attention(query, key, value, mask=None):
    """
    Compute self-attention.
    Args:
        query, key, value: Tensors of shape (batch_size, seq_len, d_k).
        mask: Tensor of shape (batch_size, seq_len, seq_len), optional.
    Returns:
        Tensor of shape (batch_size, seq_len, d_k) with attention applied.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    context = torch.matmul(attention_weights, value)
    return context, attention_weights

# Step 3: Implement Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        """
        Multi-Head Attention Mechanism.
        Args:
            num_heads: Number of attention heads.
            d_model: Dimension of the model (input embeddings).
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for Multi-Head Attention.
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, d_model).
            mask: Tensor of shape (batch_size, seq_len, seq_len), optional.
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with attention applied.
        """
        batch_size = query.size(0)

        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        context, attention_weights = self_attention(query, key, value, mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)
        return output, attention_weights

# Step 4: Define Polysemy Examples and Word Mapping
polysemy_sentences = [
    "He rose from his chair to deliver a speech",
    "The rose bloomed beautifully in the garden",
    "I need to visit the bank to deposit my check",
    "We rested on the bank of the river",
    "She decided to lead the project until completion",
    "Exposure to lead is harmful to health"
]

vocab = set(word for sentence in polysemy_sentences for word in sentence.split())
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# Step 5: Apply Multi-Head Attention to Polysemy Examples
def apply_attention(sentences, word_to_idx, d_model=64, num_heads=4):
    """
    Apply multi-head attention to polysemy examples.
    Args:
        sentences: List of tokenized sentences.
        word_to_idx: Dictionary mapping words to indices.
        d_model: Dimension of embeddings.
        num_heads: Number of attention heads.
    Returns:
        Attention weights for visualizing patterns.
    """
    vocab_size = len(word_to_idx)
    embeddings = nn.Embedding(vocab_size, d_model)

    inputs = [[word_to_idx[word] for word in sentence.split()] for sentence in sentences]
    inputs = [torch.tensor(sentence) for sentence in inputs]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)

    mask = (padded_inputs != 0).unsqueeze(1).repeat(1, padded_inputs.size(1), 1)

    mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
    outputs, attention_weights = mha(embeddings(padded_inputs), embeddings(padded_inputs), embeddings(padded_inputs), mask)
    return outputs, attention_weights

# Apply Multi-Head Attention
outputs, attention_weights = apply_attention(polysemy_sentences, word_to_idx)
print("Attention Weights Shape:", attention_weights.shape)
