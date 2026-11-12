"""Shared model registry and encoders, used by embed.py and query.py.

Both scripts must use identical model loading and preprocessing, so it lives here.
All returned embeddings are L2-normalized float32 tensors.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
import glob
import numpy as np

MODELS = {
    "clip-l14": {"checkpoint": "openai/clip-vit-large-patch14", "family": "clip"},
    "siglip-so400m": {"checkpoint": "google/siglip-so400m-patch14-384", "family": "siglip"},
    "siglip2-so400m": {"checkpoint": "google/siglip2-so400m-patch14-384", "family": "siglip2"},
    "dinov2-l14": {"checkpoint": "facebook/dinov2-large", "family": "dino"},
}


def load(name, device):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    checkpoint = MODELS[name]["checkpoint"]
    model = AutoModel.from_pretrained(checkpoint, torch_dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(checkpoint)
    return model, processor


def _features(out):
    # Some transformers versions return an output object rather than a tensor.
    return out if torch.is_tensor(out) else out.pooler_output


def preprocess_image(processor, img):
    """PIL image -> pixel tensor (C, H, W). Runs in DataLoader workers."""
    return processor(images=img, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def encode_pixels(model, pixel_values):
    pixel_values = pixel_values.to(model.dtype)
    if hasattr(model, "get_image_features"):
        feats = _features(model.get_image_features(pixel_values=pixel_values))
    else:  # vision-only backbone (DINOv2): CLS token after the final layer norm
        feats = model(pixel_values=pixel_values).pooler_output
    return F.normalize(feats.float(), dim=-1)

@torch.no_grad()
def encode_texts(name, model, processor, texts, device):
    family = MODELS[name]["family"]
    if family == "siglip2":
        texts = [t.lower() for t in texts]  # SigLIP 2 was trained on lowercased text
    if family == "clip":
        inputs = processor(text=texts, padding=True, return_tensors="pt")
    else:
        inputs = processor(text=texts, padding="max_length", max_length=64, return_tensors="pt")
    feats = _features(model.get_text_features(**inputs.to(device)))
    return F.normalize(feats.float(), dim=-1)


def score_params(model):
    """logit_scale for CLIP and SigLIP; logit_bias for SigLIP only (None for CLIP)."""
    if not hasattr(model, "logit_scale"):
        return None, None  # vision-only model: no text side
    scale = model.logit_scale.exp().item()
    bias = model.logit_bias.item() if hasattr(model, "logit_bias") else None
    return scale, bias

def load_embeddings(model_dir):
    """All shards in model_dir, concatenated in row order, as float32 (N, D)."""
    shards = sorted(glob.glob(model_dir + "/emb_*.npy"))
    return np.concatenate([np.load(s) for s in shards]).astype(np.float32)