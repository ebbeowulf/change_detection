import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class CachedTokenDataset(Dataset):
    """Returns (before tokens, after tokens, per-patch delta, label) from the cached .npz files."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        z = np.load(s['feat_path'])
        return (torch.from_numpy(z['before'].astype(np.float32)), torch.from_numpy(z['after'].astype(np.float32)),
                torch.from_numpy(z['delta'].astype(np.float32)), torch.tensor(float(s['label'])))


class SiameseHead(nn.Module):
    """Compares the before and after CLS embeddings: [b, a, |a-b|, a*b] -> MLP -> logit."""

    def __init__(self, dim, hidden=256, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(4 * dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, before, after, delta):
        b, a = self.norm(before[:, 0]), self.norm(after[:, 0])
        return self.mlp(torch.cat([b, a, (a - b).abs(), a * b], dim=1)).squeeze(1)


class DeltaPatchHead(nn.Module):
    """Compares before/after patch tokens location by location ([b, a, a-b, cosine, delta] -> shared MLP),
    then pools patches with attention whose logits are biased by log(delta), so pooling starts out
    focused where the segmentation score rose and learns how far to trust that."""

    def __init__(self, dim, n_patches, hidden=128, dropout=0.3):
        super().__init__()
        self.n_patches = n_patches
        self.norm = nn.LayerNorm(dim)
        self.patch_mlp = nn.Sequential(nn.Linear(3 * dim + 2, hidden), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(hidden, hidden), nn.ReLU())
        self.score = nn.Linear(hidden, 1)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, before, after, delta):
        b = self.norm(before[:, -self.n_patches:])
        a = self.norm(after[:, -self.n_patches:])
        cos = F.cosine_similarity(a, b, dim=-1).unsqueeze(-1)
        h = self.patch_mlp(torch.cat([b, a, a - b, cos, delta.unsqueeze(-1)], dim=-1))
        logits = self.score(h).squeeze(-1) + self.alpha * torch.log(delta + 1e-3)
        w = torch.softmax(logits, dim=1)
        return self.classifier((w.unsqueeze(-1) * h).sum(dim=1)).squeeze(1)


def build_head(name, dim, n_patches):
    if name == 'siamese':
        return SiameseHead(dim)
    if name == 'patch':
        return DeltaPatchHead(dim, n_patches)
    raise ValueError(f"unknown head {name}")