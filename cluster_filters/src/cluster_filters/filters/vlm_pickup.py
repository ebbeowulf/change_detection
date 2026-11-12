"""
VLMPickupFilter: ask the VLM whether the robot should pick up the object, once per cluster.

For each cluster:
  1. Rank its active views: by the scores of the last earlier stage that scored views (siamese probability,
     vlm_change answer), most points in frame breaking ties; or by points in frame alone if no stage scored views.
  2. Take the top --vlm_pickup_queries x --vlm_pickup_images_per_query views and deal them round-robin into
     --vlm_pickup_queries queries, so every query gets a mix of the best views and no view is sent twice.
  3. Each query sends its views together: full AFTER images with the cluster's box drawn in blue, scaled by
     --vlm_pickup_image_scale, with the prompt below. The answer is True, False, or unparseable (no answer).
  4. Cluster score = fraction of True among the answered queries; passes if >= --vlm_pickup_threshold.
     A cluster with no answered query is rejected.
Answers are cached per query (cluster, its views and boxes, model name) and saved as each one arrives.
--vlm_pickup_model NAME uses that model's saved answers whatever the server is running; queries without a saved
answer are asked only if the server runs NAME, otherwise they are left unanswered (counted and reported).
"""
import os
import numpy as np
import cv2
from PIL import Image
from .base import ClusterFilter, ScoreCache
from records import active_views

PICKUP_PROMPT = ("We are deciding on tasks for a robot that can pick small stuff up and put them away. "
                 "The robot should pick up things left behind by people that used the room. "
                 "This includes all small stuff that does not normally belong in this kind of room. "
                 "The robot should not pickup decorations or expensive electronics. "
                 "The object surrounded by the blue box in the provided image(s) has been identified by the robot "
                 "as a candidate for picking up. "
                 "Is the object surrounded by the blue box in these images an object that the robot should pick up "
                 "and put away? ")

BOX_COLOR_RGB = (0, 0, 255)       # blue
BOX_THICKNESS = 5


class VLMPickupFilter(ClusterFilter):
    name = 'vlm_pickup'

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--vlm_pickup_images_per_query', type=int, default=4, help='views sent together in one query (default=4)')
        parser.add_argument('--vlm_pickup_queries', type=int, default=3, help='queries per cluster, each with different views (default=3)')
        parser.add_argument('--vlm_pickup_threshold', type=float, default=0.5, help='fraction of answered queries that must say True (default=0.5)')
        parser.add_argument('--vlm_pickup_image_scale', type=float, default=0.5, help='scale of each image sent to the VLM (default=0.5)')
        parser.add_argument('--vlm_pickup_model', type=str, default='', help="use this model's saved answers, e.g. qwen3.5:35b+nothink+json; unsaved queries are asked only if the server runs this model (default = the server's model)")

    def __init__(self, args, run_settings, config_dir):
        super().__init__(args, run_settings, config_dir)
        from llm_query_tools import single_level_results_template, send_data_to_llm, get_model_name
        self.send, self.get_model_name = send_data_to_llm, get_model_name
        self.template = single_level_results_template(["is_pickup"], [bool], ["<True/False>"])
        self.prompt = PICKUP_PROMPT + "Return an answer in JSON format as " + self.template.generate_format_prompt()
        self.per_query = args.vlm_pickup_images_per_query
        self.n_queries = args.vlm_pickup_queries
        self.threshold = args.vlm_pickup_threshold
        self.scale = args.vlm_pickup_image_scale
        self.requested_model = args.vlm_pickup_model
        self.model_name = self.requested_model if self.requested_model else self.get_model_name()
        self.server_ok = None                      # checked once, only if a query has to be asked
        self.not_asked = 0
        self.cache = ScoreCache(config_dir + "/filters/_cache/vlm_pickup__" + self.model_name.replace('/', '_').replace(':', '_') + ".json")
        self.settings = {'model': self.model_name, 'model_from': 'command line' if self.requested_model else 'server',
                         'prompt': self.prompt, 'images_per_query': self.per_query, 'queries': self.n_queries,
                         'threshold': self.threshold, 'image_scale': self.scale}

    # ---------------------------------------------------------- choosing views

    @staticmethod
    def rank_views(record):
        """Active views, best first: by the last earlier stage's view scores, then by points in frame."""
        scores = {}
        for st in record['stages'].values():
            scores = st.get('view_scores', scores)
        return sorted(active_views(record), key=lambda v: (-scores.get(v['frame_id'], 0.0), -v.get('points_in_frame', 0)))

    def query_groups(self, record):
        """Round-robin the top-ranked views into the queries; queries left empty are dropped."""
        ranked = self.rank_views(record)[:self.per_query * self.n_queries]
        groups = [ranked[q::self.n_queries] for q in range(self.n_queries)]
        return [g for g in groups if g]

    # ---------------------------------------------------------- asking

    def can_ask(self):
        if not self.requested_model:
            return True
        if self.server_ok is None:
            try:
                self.server_ok = self.get_model_name() == self.requested_model
            except OSError:
                self.server_ok = False
        return self.server_ok

    def image_with_box(self, view):
        rs = self.run_settings
        img = np.array(Image.open(rs['root_dir'] + "/" + rs['inputs']['color_dir'] + "/" + view['image']).convert('RGB'))
        x0, y0, x1, y1 = [int(b) for b in view['bbox']]
        img = cv2.rectangle(np.ascontiguousarray(img), (x0, y0), (x1, y1), BOX_COLOR_RGB, BOX_THICKNESS)
        if self.scale < 1.0:
            img = cv2.resize(img, (0, 0), fx=self.scale, fy=self.scale)
        return img[:, :, ::-1].copy()              # send_data_to_llm expects BGR (it converts to RGB)

    def ask(self, group):
        try:
            images = [self.image_with_box(v) for v in group]
        except FileNotFoundError:
            return None
        reply = self.send(self.prompt, images)
        return self.template.recover_json(reply).get('is_pickup')

    @staticmethod
    def query_key(record, group):
        return record['uid'] + "|" + ";".join(v['frame_id'] + ":" + ",".join(str(int(b)) for b in v['bbox']) for v in group)

    def apply(self, record):
        queries = []
        for group in self.query_groups(record):
            key = self.query_key(record, group)
            answer = self.cache.scores.get(key)
            if answer is None:
                if self.can_ask():
                    answer = self.ask(group)
                    if answer is not None:
                        self.cache.scores[key] = answer
                        self.cache.changed = True
                        self.cache.save()          # saved as it arrives, so a stopped run resumes here
                else:
                    self.not_asked += 1
            queries.append({'views': [v['frame_id'] for v in group], 'answer': answer})

        answered = [q['answer'] for q in queries if q['answer'] is not None]
        score = sum(1 for a in answered if a) / len(answered) if answered else 0.0
        decision = bool(answered) and score >= self.threshold
        self.store(record, {'queries': queries, 'queries_answered': len(answered),
                            'cluster_score': score, 'decision': decision})
        return decision

    def finish(self):
        self.cache.save()
        if self.not_asked:
            print(f"  WARNING: {self.not_asked} queries had no saved {self.model_name} answer and the server is not "
                  f"running that model - left unanswered")