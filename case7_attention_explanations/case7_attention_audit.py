from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD = '<PAD>'
UNK = '<UNK>'


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def simple_tokenize(text: str) -> list[str]:
    # Keep punctuation as visible tokens so the explanation display matches the input closely.
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", str(text).lower(), flags=re.UNICODE)


def build_vocab(texts: Iterable[str], min_freq: int = 2, max_size: int = 20000) -> dict[str, int]:
    counts = Counter()
    for text in texts:
        counts.update(simple_tokenize(text))
    items = [(tok, n) for tok, n in counts.items() if n >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    vocab = {PAD: 0, UNK: 1}
    for tok, _ in items[: max_size - len(vocab)]:
        vocab[tok] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int = 80) -> tuple[list[int], list[str]]:
    tokens = simple_tokenize(text)[:max_len]
    if not tokens:
        raise ValueError('Input must contain at least one token.')
    ids = [vocab.get(t, vocab[UNK]) for t in tokens]
    return ids, tokens


class SSTDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: dict[str, int], max_len: int = 80):
        self.frame = frame.reset_index(drop=True).copy()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        ids, tokens = encode_text(row['sentence'], self.vocab, self.max_len)
        return {'ids': ids, 'tokens': tokens, 'sentence': str(row['sentence']), 'label': int(row['label']), 'idx': int(row.get('idx', idx))}


def make_collate(pad_id: int = 0):
    def collate(batch):
        lengths = torch.tensor([max(1, len(x['ids'])) for x in batch], dtype=torch.long)
        max_len = int(lengths.max())
        ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, item in enumerate(batch):
            seq = item['ids'] or [pad_id]
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, :len(seq)] = True
        labels = torch.tensor([x['label'] for x in batch], dtype=torch.long)
        return ids, lengths, mask, labels, batch
    return collate


class BiLSTMAttention(nn.Module):
    """Small, inspectable sequence classifier with one additive attention distribution."""
    def __init__(self, vocab_size: int, emb_dim: int = 96, hidden_dim: int = 96, dropout: float = 0.25, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def encode(self, ids, lengths):
        emb = self.embedding(ids)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_h, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(packed_h, batch_first=True, total_length=ids.size(1))
        return emb, hidden

    def attention_from_hidden(self, hidden, mask):
        score = self.attn_score(torch.tanh(self.attn_proj(hidden))).squeeze(-1)
        score = score.masked_fill(~mask, float('-inf'))
        return torch.softmax(score, dim=-1)

    def classify_from_hidden(self, hidden, mask, attention_override=None):
        if attention_override is None:
            attention = self.attention_from_hidden(hidden, mask)
        else:
            attention = attention_override.masked_fill(~mask, 0.0)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        context = torch.sum(hidden * attention.unsqueeze(-1), dim=1)
        logits = self.classifier(self.dropout(context))
        return logits, attention

    def forward(self, ids, lengths, mask, attention_override=None, return_hidden=False):
        emb, hidden = self.encode(ids, lengths)
        logits, attention = self.classify_from_hidden(hidden, mask, attention_override)
        if return_hidden:
            return logits, attention, emb, hidden
        return logits, attention


def train_one_epoch(model, loader, optimizer, device='cpu'):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total, correct = 0.0, 0, 0
    for ids, lengths, mask, labels, _ in loader:
        ids, lengths, mask, labels = ids.to(device), lengths.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(ids, lengths, mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        total += len(labels)
    return {'loss': total_loss / total, 'accuracy': correct / total}


@torch.no_grad()
def evaluate(model, loader, device='cpu') -> pd.DataFrame:
    model.eval()
    rows = []
    for ids, lengths, mask, labels, raw in loader:
        ids, lengths, mask = ids.to(device), lengths.to(device), mask.to(device)
        logits, attn = model(ids, lengths, mask)
        probs = torch.softmax(logits, -1)
        pred = probs.argmax(1)
        for j, item in enumerate(raw):
            n = int(mask[j].sum())
            rows.append({
                'idx': item['idx'], 'sentence': item['sentence'], 'label': int(labels[j]),
                'prediction': int(pred[j]), 'p_positive': float(probs[j, 1]),
                'confidence': float(probs[j, pred[j]]),
                'correct': bool(pred[j].item() == labels[j].item()),
                'tokens': item['tokens'], 'attention': attn[j, :n].cpu().numpy().tolist(),
            })
    return pd.DataFrame(rows)


def prediction_details(model, ids, lengths, mask, target_class=None):
    model.eval()
    logits, attention, emb, hidden = model(ids, lengths, mask, return_hidden=True)
    probs = torch.softmax(logits, -1)
    pred = logits.argmax(-1)
    if target_class is None:
        target_class = int(pred.item())
    return logits, probs, attention, emb, hidden, target_class


def gradient_importance(model, ids, lengths, mask, target_class=None) -> np.ndarray:
    """L1 magnitude of gradient x embedding for the target-vs-other logit margin.

    Normalized across valid positions. This unsigned diagnostic loses direction;
    it is neither an additive explanation nor a ground-truth importance label.
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    emb = model.embedding(ids)
    emb.retain_grad()
    packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
    packed_h, _ = model.encoder(packed)
    hidden, _ = pad_packed_sequence(packed_h, batch_first=True, total_length=ids.size(1))
    logits, _ = model.classify_from_hidden(hidden, mask)
    if target_class is None:
        target_class = int(logits.argmax(-1).item())
    (logits[0, target_class] - logits[0, 1-target_class]).backward()
    score = (emb.grad[0] * emb.detach()[0]).abs().sum(-1)
    score = score * mask[0]
    score = score / score.sum().clamp_min(1e-12)
    return score.detach().cpu().numpy()


def uniform_attention(mask: torch.Tensor) -> torch.Tensor:
    a = mask.float()
    return a / a.sum(-1, keepdim=True).clamp_min(1e-12)


def shuffled_attention(attention: torch.Tensor, mask: torch.Tensor, seed: int = 2026) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    out = torch.zeros_like(attention)
    for i in range(attention.size(0)):
        n = int(mask[i].sum())
        vals = attention[i, :n].detach().cpu().numpy().copy()
        rng.shuffle(vals)
        out[i, :n] = torch.tensor(vals, device=attention.device, dtype=attention.dtype)
    return out


@torch.no_grad()
def fixed_hidden_intervention(model, ids, lengths, mask, kind='uniform', seed=2026):
    model.eval()
    _, hidden = model.encode(ids, lengths)
    logits0, attn0 = model.classify_from_hidden(hidden, mask)
    if kind == 'uniform':
        attn1 = uniform_attention(mask).to(hidden.device)
    elif kind == 'shuffle':
        attn1 = shuffled_attention(attn0, mask, seed=seed)
    else:
        raise ValueError("kind must be 'uniform' or 'shuffle'")
    logits1, _ = model.classify_from_hidden(hidden, mask, attention_override=attn1)
    p0 = torch.softmax(logits0, -1)
    p1 = torch.softmax(logits1, -1)
    return {
        'p_original': p0.cpu().numpy(), 'p_perturbed': p1.cpu().numpy(),
        'attention_original': attn0.cpu().numpy(), 'attention_perturbed': attn1.cpu().numpy(),
        'delta_positive': float(abs(p0[0,1] - p1[0,1])),
        'label_changed': bool(logits0.argmax(-1).item() != logits1.argmax(-1).item()),
    }


def rankdata_average(x: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j+1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    ra, rb = rankdata_average(a), rankdata_average(b)
    return float(np.corrcoef(ra, rb)[0,1])


def topk_overlap(a: Sequence[float], b: Sequence[float], k: int = 3) -> float:
    k = min(k, len(a), len(b))
    if k <= 0:
        return float('nan')
    ia = set(np.argsort(np.asarray(a))[-k:])
    ib = set(np.argsort(np.asarray(b))[-k:])
    return len(ia & ib) / k


def dataframe_overview(df: pd.DataFrame) -> dict:
    lengths = df['sentence'].astype(str).map(lambda s: len(simple_tokenize(s)))
    counts = df['label'].value_counts().sort_index().to_dict() if 'label' in df else {}
    return {
        'rows': len(df), 'class_counts': counts,
        'token_length_min': int(lengths.min()), 'token_length_median': float(lengths.median()),
        'token_length_p95': float(lengths.quantile(.95)), 'token_length_max': int(lengths.max()),
        'duplicate_sentences': int(df['sentence'].duplicated().sum()),
    }
