"""
Turn segmentation of an AFTER image and its rendered BEFORE view into 3D points.

  pcloud_change.setup_segmentation(prompts, classifier, sam3_confidence)   load SAM3 or CLIPseg once
  pcloud_change.load_image(color, depth, pose, key)                         prepare one AFTER frame
  pcloud_change.segment_pair(prompts, before_image)                         score maps for AFTER and BEFORE
  pcloud_change.get_pts(score_map, mask)                                    masked pixels -> 3D points + scores
"""
import numpy as np
import torch
from PIL import Image
from change_pcloud_utils.map_utils import get_rotated_points, DEVICE
from change_pcloud_utils.camera_params import camera_params


class pcloud_change():
    def __init__(self, params: camera_params):
        self.params = params
        # pixel offsets from the principal point, reused for every frame
        self.rows = torch.tensor(np.tile(np.arange(params.height).reshape(params.height, 1), (1, params.width)) - params.cy, device=DEVICE)
        self.cols = torch.tensor(np.tile(np.arange(params.width), (params.height, 1)) - params.cx, device=DEVICE)
        self.rot_matrixT = torch.tensor(params.rot_matrix, device=DEVICE)
        self.loaded_image = None
        self.YS = None

    def setup_segmentation(self, prompts, classifier, sam3_confidence):
        """SAM3 (detections above sam3_confidence) or CLIPseg (per-pixel probabilities)."""
        if classifier == 'sam3':
            from segmentation_utils.sam3_segmentation import sam3_segmentation
            self.YS = sam3_segmentation(prompts, sam3_confidence)
        elif classifier == 'clipseg':
            from segmentation_utils.clip_segmentation import clip_seg
            self.YS = clip_seg(prompts)
        else:
            raise ValueError(f"unsupported classifier '{classifier}' (sam3 or clipseg)")

    def load_image(self, colorI_PIL: Image, depthI_np: np.ndarray, poseM: np.ndarray, uid_key: str, max_distance=10.0):
        """Prepare one AFTER frame: per-pixel camera coordinates, valid-depth mask (between 1e-4 m and
        max_distance), and the camera-to-map transform. Returns False if the frame could not be loaded."""
        try:
            depthT = torch.tensor(depthI_np.astype('float') / 1000.0, device=DEVICE)
            self.loaded_image = {
                'key': uid_key,
                'color': colorI_PIL,
                'depthT': depthT,
                'colorT': torch.tensor(np.array(colorI_PIL), device=DEVICE),
                'x': self.cols * depthT / self.params.fx,
                'y': self.rows * depthT / self.params.fy,
                'depth_mask': (depthT > 1e-4) * (depthT < max_distance),
                'M': torch.matmul(self.rot_matrixT, torch.tensor(poseM, device=DEVICE)),
            }
            return True
        except Exception as e:
            print(f"Failed to load image materials for {uid_key}: {e}")
            self.loaded_image = None
            return False

    def segment_pair(self, prompts, before_image: Image):
        """Score maps for every prompt on the loaded AFTER image and on before_image (its render).
        Returns {prompt: {'new_prob': tensor or None, 'baseline_prob': tensor or None}}."""
        result = {p: {'new_prob': None, 'baseline_prob': None} for p in prompts}
        self.YS.process_image(self.loaded_image['color'])
        for p in prompts:
            result[p]['new_prob'] = self.YS.get_prob_array(p)
        self.YS.process_image(before_image)
        for p in prompts:
            result[p]['baseline_prob'] = self.YS.get_prob_array(p)
        return result

    def get_pts(self, score_map, maskT):
        """3D points (map frame), colors and scores of the pixels in maskT."""
        pts_rot = get_rotated_points(self.loaded_image['x'], self.loaded_image['y'], self.loaded_image['depthT'],
                                     maskT, self.loaded_image['M'])
        return {'xyz': pts_rot, 'rgb': self.loaded_image['colorT'][maskT], 'probs': score_map[maskT]}