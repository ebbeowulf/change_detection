"""
VLMChangeFilter: the published before/after VLM filter ("before_unannotated_and_after").

For each active view of a cluster (at most --vlm_change_max_views, chosen at random with a fixed seed):
  1. BEFORE image: the robot's own earlier image (frame keyword --vlm_change_before_keyword in the run's
     colmap_combined registration) that sees the cluster centroid from the most similar position:
     similarity = exp(-0.5 (distance difference / 1.0 m)^2) x exp(-0.5 (angle between the two rays to the
     centroid / 0.25 rad)^2)
  2. AFTER image: the view's image with the cluster's box drawn in blue (the same box the siamese filter uses).
  3. The two are placed side by side (BEFORE left, AFTER right, white strip between, scaled by 0.5) and the
     VLM server is asked whether a new object is inside the blue box on the right (prompt below).
View score = 1 (True) or 0 (False); an answer that cannot be parsed gives no score (listed as missing).
Cluster score = fraction of True answers; the cluster passes if it is >= --vlm_change_cluster_threshold.
VLM answers are cached per cluster, view, box and model name, and saved as each one arrives.
--vlm_change_model NAME uses that model's saved answers whatever the server is running; pairs without a saved
answer are asked only if the server runs NAME, otherwise they are left unanswered (counted and reported).
"""
import os
import random
import numpy as np
import cv2
from PIL import Image
from .base import ViewScoringFilter, ScoreCache

# prompt of the published "before_unannotated_and_after" filter, unchanged (including its spacing)
UNANNOTATED_PROMPT = ['The attached image includes two side by side images of the same location captured at different times.',
                      'A box has been drawn in the right most image to specify the area of interest.'
                      'Please tell me if there is a new object contained inside the blue box on the right that was not present in the image on the left.'
                      'If there is a new object, answer True. If the area has not changed, answer False.']

BOX_COLOR_RGB = (0, 0, 255)       # blue
BOX_THICKNESS = 5
DIST_SIGMA_M = 1.0                # matching: camera-to-object distance difference
ANGLE_SIGMA_RAD = 0.25            # matching: angle between the rays to the centroid


class VLMChangeFilter(ViewScoringFilter):
    name = 'vlm_change'

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--vlm_change_cluster_threshold', type=float, default=0.5, help='fraction of True answers a cluster needs (default=0.5)')
        parser.add_argument('--vlm_change_max_views', type=int, default=20, help='views asked per cluster, chosen at random with a fixed seed (default=20)')
        parser.add_argument('--vlm_change_before_keyword', type=str, default='color', help="frame keyword of the BEFORE images in colmap_combined images.txt (default=color)")
        parser.add_argument('--vlm_change_image_scale', type=float, default=0.5, help='scale of the side-by-side image sent to the VLM (default=0.5)')
        parser.add_argument('--vlm_change_seed', type=int, default=0, help='seed for choosing which views are asked (default=0)')
        parser.add_argument('--vlm_change_model', type=str, default='', help='use this model\'s saved answers, e.g. qwen3.5:35b+nothink+json; uncached pairs are asked only if the server runs this model (default = the server\'s model)')

    def __init__(self, args, run_settings, config_dir):
        super().__init__(args, run_settings, config_dir)
        from llm_utils import before_and_after_cluster_filter
        self.vlm = before_and_after_cluster_filter(UNANNOTATED_PROMPT, image_scale=args.vlm_change_image_scale)
        self.requested_model = args.vlm_change_model
        self.model_name = self.requested_model if self.requested_model else self.vlm.active_model_name()
        self.not_asked = 0                         # uncached pairs skipped because the server runs another model
        self.image_threshold = 0.5                       # view scores are 0 or 1
        self.cluster_threshold = args.vlm_change_cluster_threshold
        self.max_views = args.vlm_change_max_views
        self.seed = args.vlm_change_seed
        self.load_before_frames(args.vlm_change_before_keyword, config_dir)
        self.cache = ScoreCache(config_dir + "/filters/_cache/vlm_change__" + self.model_name.replace('/', '_').replace(':', '_') + ".json")
        self.settings = {'model': self.model_name, 'model_from': 'command line' if self.requested_model else 'server',
                         'prompt': self.vlm.PROMPT, 'image_scale': args.vlm_change_image_scale,
                         'cluster_threshold': self.cluster_threshold, 'max_views': self.max_views, 'seed': self.seed,
                         'before_keyword': args.vlm_change_before_keyword, 'before_frames': len(self.fList_base.keys()),
                         'matching': {'dist_sigma_m': DIST_SIGMA_M, 'angle_sigma_rad': ANGLE_SIGMA_RAD}}

    # ---------------------------------------------------------- BEFORE images

    def load_before_frames(self, keyword, config_dir):
        """Camera parameters and the BEFORE frames, read the same way step 3 reads the AFTER frames."""
        from change_pcloud_utils.colmap_utils import get_camera_params, build_file_list
        rs = self.run_settings
        colmap_dir = rs['root_dir'] + "/" + rs['inputs']['colmap_dir'] + "/"
        color_dir = rs['root_dir'] + "/" + rs['inputs']['color_dir'] + "/"
        depth_dir = rs['root_dir'] + "/" + rs['inputs']['depth_dir'] + "/"
        work_dir = config_dir + "/filters/_cache"
        os.makedirs(work_dir, exist_ok=True)
        self.params = get_camera_params(colmap_dir, rs['nerfacto_dir'])
        self.fList_base = build_file_list(color_dir, depth_dir, work_dir, colmap_dir, keyword)
        if len(self.fList_base.keys()) == 0:
            raise SystemExit(f"ERROR: no '{keyword}' frames in {colmap_dir}images.txt - check --vlm_change_before_keyword")
        self.after_poses = None

    def camera_center(self, fList, key):
        return np.matmul(self.params.rot_matrix, fList.get_pose(key))[:3, 3]

    def best_before_frame(self, centroid, after_center, before_keys):
        """BEFORE frame seeing the centroid from the most similar distance and direction as the AFTER camera."""
        after_ray = centroid - after_center
        after_dist = np.linalg.norm(after_ray)
        best, best_sim = None, -1.0
        for key in before_keys:
            ray = centroid - self.camera_center(self.fList_base, key)
            dist = np.linalg.norm(ray)
            angle = np.arccos(np.clip(np.dot(ray / dist, after_ray / after_dist), -1.0, 1.0))
            sim = np.exp(-0.5 * ((dist - after_dist) / DIST_SIGMA_M) ** 2) * np.exp(-0.5 * (angle / ANGLE_SIGMA_RAD) ** 2)
            if sim > best_sim:
                best, best_sim = key, sim
        return best

    # ---------------------------------------------------------- scoring

    def after_center(self, view):
        """AFTER camera center for a view, from the run's own registration (AFTER frames, keyword 'new')."""
        if self.after_poses is None:
            from change_pcloud_utils.colmap_utils import build_file_list
            rs = self.run_settings
            self.after_poses = build_file_list(rs['root_dir'] + "/" + rs['inputs']['color_dir'] + "/",
                                               rs['root_dir'] + "/" + rs['inputs']['depth_dir'] + "/",
                                               self.config_dir + "/filters/_cache",
                                               rs['root_dir'] + "/" + rs['inputs']['colmap_dir'] + "/",
                                               rs['inputs']['frame_keyword'])
        return self.camera_center(self.after_poses, int(view['pose_key']))

    def select_views(self, record, views):
        """At most max_views views, the same ones every time this cluster is filtered."""
        if len(views) <= self.max_views:
            return views
        return random.Random(f"{self.seed}-{record['uid']}").sample(views, self.max_views)

    def server_runs_model(self):
        try:
            return self.vlm.active_model_name() == self.requested_model
        except OSError:                            # server not running
            return False

    def score_views(self, record, views):
        from change_pcloud_utils.map_utils import identify_related_images_global_pose
        rs = self.run_settings
        if self.requested_model and not self.server_runs_model():
            self.not_asked += len(views)
            return {}                              # only saved answers can be used; these pairs stay unanswered
        centroid = np.asarray(record['centroid'], dtype=float)[:3]
        before_keys = identify_related_images_global_pose(self.params, self.fList_base, centroid)
        if not before_keys:
            return {}                                     # no BEFORE image sees this cluster

        scores = {}
        for v in views:
            before_key = self.best_before_frame(centroid, self.after_center(v), before_keys)
            try:
                after = np.array(Image.open(rs['root_dir'] + "/" + rs['inputs']['color_dir'] + "/" + v['image']).convert('RGB'))
                before = Image.open(self.fList_base.get_color_fileName(before_key)).convert('RGB')
            except FileNotFoundError:
                continue
            if before.size != (after.shape[1], after.shape[0]):
                before = before.resize((after.shape[1], after.shape[0]), Image.BILINEAR)
            x0, y0, x1, y1 = [int(b) for b in v['bbox']]
            after = cv2.rectangle(np.ascontiguousarray(after), (x0, y0), (x1, y1), BOX_COLOR_RGB, BOX_THICKNESS)
            answer = self.vlm.evaluate_image_pair(np.array(before), after).get('is_new_object')
            if answer is not None:
                scores[v['frame_id']] = 1.0 if answer else 0.0
                self.cache.put(record['uid'], v, scores[v['frame_id']])   # save each answer as it arrives,
                self.cache.save()                                          # so a stopped run resumes from here
            v['vlm_change_before_frame'] = str(before_key)
        return scores

    def finish(self):
        super().finish()
        if self.not_asked:
            print(f"  WARNING: {self.not_asked} pairs had no saved {self.model_name} answer and the server is not "
                  f"running that model - left unanswered")