"""
Base classes for the filters.

ClusterFilter        every filter. The runner calls apply(record) on each still-active cluster, in order.
ViewScoringFilter    scores each active view, then decides the cluster: the fraction of views scoring
                     >= image_threshold must be >= cluster_threshold. Children implement score_views().
ViewEditFilter       edits views (marks them inactive or replaces their box); rejects a cluster only when
                     no active views remain. Children implement edit_view().
ScoreCache           per-view scores of one filter + model, reused across reruns and reordered sequences.
"""
import json
import os
from records import active_views


class ClusterFilter:
    """Contract for every filter.
        name        used in --stages and as the key in each record's 'stages'
        add_args    classmethod adding this filter's own options, prefixed with its name
        __init__    receives the parsed args, the step-3 run_settings.json contents and config_dir
        settings    dict written into results.json (models, thresholds, where they came from)
        apply       record -> True (keep) / False (reject); stores its details via store()
        finish      called once after the last cluster (save caches, free models)"""
    name = None

    @classmethod
    def add_args(cls, parser):
        pass

    def __init__(self, args, run_settings, config_dir):
        self.run_settings = run_settings
        self.config_dir = config_dir
        self.settings = {}

    def apply(self, record):
        raise NotImplementedError

    def store(self, record, details):
        record['stages'][self.name] = details

    def finish(self):
        pass


class ScoreCache:
    """Per-view scores keyed by cluster, frame and box, stored in <config_dir>/filters/_cache/<file>."""

    def __init__(self, path):
        self.path = path
        self.scores = {}
        if os.path.exists(path):
            with open(path) as f:
                self.scores = json.load(f)
        self.changed = False

    @staticmethod
    def key(uid, view):
        return uid + "|" + view['frame_id'] + "|" + ",".join(str(int(b)) for b in view['bbox'])

    def get(self, uid, view):
        return self.scores.get(self.key(uid, view))

    def put(self, uid, view, score):
        self.scores[self.key(uid, view)] = score
        self.changed = True

    def save(self):
        if self.changed:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(self.scores, f)
            os.replace(tmp, self.path)
            self.changed = False


class ViewScoringFilter(ClusterFilter):
    """Score every active view, then decide the cluster by the fraction of views at or above the image
    threshold. Children set self.image_threshold, self.cluster_threshold and self.cache (a ScoreCache)
    in __init__, and implement score_views(record, views) -> {frame_id: score}, returning no entry for a
    view whose inputs are missing. Children may override select_views() to score only some views
    (e.g. a capped random sample); views not selected are recorded as 'views_not_selected'."""

    def score_views(self, record, views):
        raise NotImplementedError

    def select_views(self, record, views):
        return views

    def apply(self, record):
        all_active = active_views(record)
        views = self.select_views(record, all_active)
        selected = {v['frame_id'] for v in views}
        view_scores, todo = {}, []
        for v in views:                                    # cached scores first
            cached = self.cache.get(record['uid'], v)
            if cached is None:
                todo.append(v)
            else:
                view_scores[v['frame_id']] = cached
        if todo:
            new = self.score_views(record, todo)
            for v in todo:
                if v['frame_id'] in new:
                    view_scores[v['frame_id']] = new[v['frame_id']]
                    self.cache.put(record['uid'], v, new[v['frame_id']])

        n_scored = len(view_scores)
        n_changed = sum(1 for s in view_scores.values() if s >= self.image_threshold)
        cluster_score = n_changed / n_scored if n_scored else 0.0
        decision = n_scored > 0 and cluster_score >= self.cluster_threshold
        self.store(record, {'view_scores': view_scores, 'views_scored': n_scored, 'views_changed': n_changed,
                            'views_missing_inputs': [v['frame_id'] for v in views if v['frame_id'] not in view_scores],
                            'views_not_selected': [v['frame_id'] for v in all_active if v['frame_id'] not in selected],
                            'cluster_score': cluster_score, 'decision': decision})
        return decision

    def finish(self):
        self.cache.save()


class ViewEditFilter(ClusterFilter):
    """Edit each active view: edit_view(record, view) returns None to drop the view (marked inactive), or a
    new box [x0, y0, x1, y1] to replace its box (the original is kept as 'original_bbox'), or the view's own
    box to leave it unchanged. The cluster is rejected only when no active views remain."""

    def edit_view(self, record, view):
        raise NotImplementedError

    def apply(self, record):
        dropped, reboxed = [], []
        for v in active_views(record):
            result = self.edit_view(record, v)
            if result is None:
                v['active'] = False
                dropped.append(v['frame_id'])
            elif list(result) != list(v['bbox']):
                v.setdefault('original_bbox', v['bbox'])
                v['bbox'] = [int(b) for b in result]
                reboxed.append(v['frame_id'])
        remaining = len(active_views(record))
        self.store(record, {'views_dropped': dropped, 'views_reboxed': reboxed, 'views_remaining': remaining,
                            'decision': remaining > 0})
        return remaining > 0