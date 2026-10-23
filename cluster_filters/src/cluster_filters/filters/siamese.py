"""
SiameseFilter: the trained change filter (DINOv2 siamese or patch head, or the delta ResNet), applied to
every active view.

--siamese_model is a training output folder (train_change_filter.py) holding:
    best.pt          weights
    train_log.json   model type and image settings (metrics.json for models from the older trainer)
    thresholds.json  image and cluster thresholds learned on val (select_thresholds.py)
View score = the model's probability that the view shows a real change.
"""
import json
import os
import numpy as np
from PIL import Image
from .base import ViewScoringFilter, ScoreCache


def prompt_slug(prompt):
    return prompt.strip().lower().replace(' ', '_')


class SiameseFilter(ViewScoringFilter):
    name = 'siamese'

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--siamese_model', type=str, default='', help='training output folder holding best.pt, train_log.json and thresholds.json')
        parser.add_argument('--siamese_image_threshold', type=float, default=-1.0, help='override the image threshold from thresholds.json (default=-1, use the file)')
        parser.add_argument('--siamese_cluster_threshold', type=float, default=-1.0, help='override the cluster threshold from thresholds.json (default=-1, use the file)')
        parser.add_argument('--siamese_batch_size', type=int, default=32, help='views per forward pass (default=32)')

    def __init__(self, args, run_settings, config_dir):
        super().__init__(args, run_settings, config_dir)
        import torch
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = args.siamese_batch_size
        model_dir = args.siamese_model.rstrip('/')
        if not model_dir or not os.path.exists(model_dir + "/best.pt"):
            raise SystemExit(f"ERROR: --siamese_model must be a training output folder with best.pt (got '{model_dir}')")

        self.load_training_settings(model_dir)
        self.load_thresholds(model_dir, args)
        self.build_model(model_dir)
        self.cache = ScoreCache(config_dir + "/filters/_cache/siamese__" + os.path.basename(os.path.abspath(model_dir)) + ".json")
        self.settings = {'model_dir': os.path.abspath(model_dir), 'model_type': self.kind, 'img_size': self.img_size,
                         'gray': self.gray, 'dino_model': self.dino_model,
                         'image_threshold': self.image_threshold, 'cluster_threshold': self.cluster_threshold,
                         'thresholds_from': self.thresholds_from}

    # ---------------------------------------------------------- setup

    def load_training_settings(self, model_dir):
        """Model type and image settings, exactly as used in training."""
        log_path = model_dir + "/train_log.json"
        if not os.path.exists(log_path):
            log_path = model_dir + "/metrics.json"
        with open(log_path) as f:
            train_args = json.load(f)['args']
        self.kind = train_args.get('model', train_args.get('head'))
        self.img_size = train_args.get('img_size', 224)
        self.gray = bool(train_args.get('gray', 1))
        self.dino_model = train_args.get('dino_model', 'facebook/dinov2-small')

    def load_thresholds(self, model_dir, args):
        """Thresholds learned on val (thresholds.json), unless overridden on the command line."""
        learned = {}
        if os.path.exists(model_dir + "/thresholds.json"):
            with open(model_dir + "/thresholds.json") as f:
                learned = json.load(f)
        override = args.siamese_image_threshold >= 0 or args.siamese_cluster_threshold >= 0
        self.image_threshold = args.siamese_image_threshold if args.siamese_image_threshold >= 0 else learned.get('image_threshold')
        self.cluster_threshold = args.siamese_cluster_threshold if args.siamese_cluster_threshold >= 0 else learned.get('cluster_threshold')
        if self.image_threshold is None or self.cluster_threshold is None:
            raise SystemExit(f"ERROR: no thresholds.json in {model_dir} - run select_thresholds.py on its "
                             "val_predictions.csv, or give --siamese_image_threshold and --siamese_cluster_threshold")
        self.thresholds_from = 'command line' if override else os.path.abspath(model_dir + "/thresholds.json")

    def build_model(self, model_dir):
        torch = self.torch
        if self.kind == 'resnet':
            from train_change_filter import ResNetFilter
            self.model = ResNetFilter()
        elif self.kind in ('siamese', 'patch'):
            from transformers import AutoModel
            from dino_heads import build_head
            self.dino = AutoModel.from_pretrained(self.dino_model).to(self.device).eval()
            self.model = build_head(self.kind, self.dino.config.hidden_size, (self.img_size // 14) ** 2)
        else:
            raise SystemExit(f"ERROR: unknown model type '{self.kind}' in {model_dir}")
        self.model.load_state_dict(torch.load(model_dir + "/best.pt", map_location=self.device))
        self.model.to(self.device).eval()

    # ---------------------------------------------------------- scoring

    def view_inputs(self, record, view):
        """Model inputs for one view, built exactly as in training; None if an image or delta is missing.
        resnet: (delta crop,)    siamese / patch: (before crop, after crop, per-patch delta)"""
        from delta_dataset import square_crop
        from dino_features import to_input, delta_patch_map
        rs = self.run_settings
        after_path = rs['root_dir'] + "/" + rs['inputs']['color_dir'] + "/" + view['image']
        render_path = rs['root_dir'] + "/" + rs['inputs']['renders_dir'] + "/rgb_" + view['frame_id'] + ".png"
        delta_path = rs['delta_dir'] + "/" + prompt_slug(record['prompt']) + "_" + view['frame_id'] + ".npy"
        try:
            delta = np.load(delta_path).astype(np.float32) if self.kind in ('resnet', 'patch') else None
            if self.kind == 'resnet':
                return (square_crop(delta, view['bbox'], self.img_size)[None],)
            a_img = Image.open(after_path).convert('RGB')
            b_img = Image.open(render_path).convert('RGB')
            if b_img.size != a_img.size:
                b_img = b_img.resize(a_img.size, Image.BILINEAR)
            a = to_input(square_crop(np.asarray(a_img, dtype=np.float32), view['bbox'], self.img_size), self.gray)
            b = to_input(square_crop(np.asarray(b_img, dtype=np.float32), view['bbox'], self.img_size), self.gray)
            d = (delta_patch_map(square_crop(delta, view['bbox'], self.img_size), self.img_size) if delta is not None
                 else np.zeros((self.img_size // 14) ** 2, np.float32))
            return b, a, d
        except FileNotFoundError:
            return None

    def forward(self, batch):
        """Probabilities for a batch of view inputs."""
        torch = self.torch
        if self.kind == 'resnet':
            logits = self.model(torch.from_numpy(np.stack([b[0] for b in batch])).to(self.device))
        else:
            n = len(batch)
            pix = torch.from_numpy(np.stack([b[0] for b in batch] + [b[1] for b in batch])).to(self.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.device == 'cuda')):
                tokens = self.dino(pixel_values=pix).last_hidden_state.float()
            delta = torch.from_numpy(np.stack([b[2] for b in batch])).to(self.device)
            logits = self.model(tokens[:n], tokens[n:], delta)
        return torch.sigmoid(logits).float().cpu().tolist()

    def score_views(self, record, views):
        ready = [(v, self.view_inputs(record, v)) for v in views]
        ready = [(v, x) for v, x in ready if x is not None]
        scores = {}
        with self.torch.no_grad():
            for start in range(0, len(ready), self.batch_size):
                batch = ready[start:start + self.batch_size]
                for (v, _), p in zip(batch, self.forward([x for _, x in batch])):
                    scores[v['frame_id']] = p
        return scores