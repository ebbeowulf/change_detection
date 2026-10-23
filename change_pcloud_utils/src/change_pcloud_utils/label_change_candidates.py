"""
Label the clusters written by build_change_candidates.py, several clusters per page, by clicking.

  python -m change_pcloud_utils.label_change_candidates <run>/change2.0_results/sam3_0.1_openVocab

Each page shows --per_page clusters of one prompt. Each cluster is a tile of --views_per_tile views,
cropped around the cluster's box (the same views every time the cluster is shown).

  left click          toggle CHANGED          (green frame)
  right click         toggle CAN BE PICKED UP (blue frame + "pickup")
  middle click or e   expand: show this cluster's views (up to --max_expand) in place of the page;
                      left/right clicks work there too; middle click, e or Esc returns to the page
  space or ->         save this page and go to the next (clusters not clicked are saved as NOT changed)
  <-                  previous page (to fix labels; saved again when you move forward)
  t                   switch the page between AFTER images and BEFORE renders
  q                   save the pages already moved past and quit (the current page is not saved)

Labels go to <config_dir>/<prompt>.labels.json, keyed by cluster ID:
    {"small_items/0007": {"changed": true, "pickup": false, "name": ""}, ...}
"name" is "unchanged" for unchanged clusters and "" for changed ones, to be filled in by a later naming pass.
Labeling resumes at the first page with an unlabeled cluster. Clusters with no views are listed and skipped.
Nothing outside config_dir is written.
"""
import argparse
import glob
import json
import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

for key in ('keymap.quit', 'keymap.back', 'keymap.forward', 'keymap.home', 'keymap.save', 'keymap.fullscreen'):
    plt.rcParams[key] = []          # these keys are used for labeling instead

CHANGED_COLOR, PICKUP_COLOR, PLAIN_COLOR = 'lime', 'deepskyblue', '0.6'
PANEL_FILLS = ('#dde6f2', '#f2e6d6')        # alternating tile backgrounds so neighboring clusters stand apart


def prompt_slug(prompt):
    return prompt.strip().lower().replace(' ', '_')


def load_clusters(config_dir, prompt):
    records = []
    for path in sorted(glob.glob(config_dir + "/clusters/" + prompt_slug(prompt) + "_[0-9][0-9][0-9][0-9].json")):
        with open(path) as f:
            records.append(json.load(f))
    return records


def load_labels(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_labels(path, labels):
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(labels, f, indent=1)
    os.replace(tmp, path)


def pick_views(record, n, seed):
    """n views, chosen the same way every time this cluster is shown."""
    views = list(record['views'])
    if len(views) > n:
        views = random.Random(f"{seed}-{record['uid']}").sample(views, n)
    return sorted(views, key=lambda v: v['frame_id'])


def crop_around_box(img, bbox, context, display_px):
    """Square crop centered on the box, side = context x the longer box side (at least 96 px), clipped to the
    image, then shrunk to at most display_px on a side (keeps drawing fast over a remote display).
    Returns the crop and the box in crop coordinates."""
    x0, y0, x1, y1 = bbox
    side = max(96, context * max(x1 - x0, y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = img.size
    cx0, cy0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    cx1, cy1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = img.crop((cx0, cy0, cx1, cy1))
    scale = min(1.0, display_px / max(crop.size))
    if scale < 1.0:
        crop = crop.resize((max(1, int(crop.size[0] * scale)), max(1, int(crop.size[1] * scale))), Image.BILINEAR)
    return np.asarray(crop), tuple(v * scale for v in (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0))


class Labeler:
    def __init__(self, config_dir, settings, prompts, args):
        self.args = args
        self.config_dir = config_dir
        self.root = settings['root_dir']
        self.inputs = settings['inputs']
        self.show_before = False
        self.labels = {}                     # prompt -> saved labels
        self.pending = {}                    # uid -> {'changed', 'pickup'} while on screen
        self.pages = []                      # (prompt, [records]) ; pages never mix prompts
        self.no_views = []
        for p in prompts:
            self.labels[p] = load_labels(self.labels_path(p))
            records = []
            for r in load_clusters(config_dir, p):
                if r['views']:
                    records.append(r)
                else:
                    self.no_views.append(r['uid'])
            for i in range(0, len(records), args.per_page):
                self.pages.append((p, records[i:i + args.per_page]))
        # start at the first page with an unlabeled cluster
        self.page = next((i for i, (p, recs) in enumerate(self.pages)
                          if any(r['uid'] not in self.labels[p] for r in recs)), len(self.pages))
        self.axes_owner = {}                 # axes -> record, for click lookup
        self.expanded = None                 # record shown in place of the page, or None

    def labels_path(self, prompt):
        return self.config_dir + "/" + prompt_slug(prompt) + ".labels.json"

    def state(self, prompt, uid):
        if uid not in self.pending:
            saved = self.labels[prompt].get(uid, {})
            self.pending[uid] = {'changed': saved.get('changed', False), 'pickup': saved.get('pickup', False)}
        return self.pending[uid]

    def image_path(self, view):
        if self.show_before:
            return self.root + "/" + self.inputs['renders_dir'] + "/rgb_" + view['frame_id'] + ".png"
        return self.root + "/" + self.inputs['color_dir'] + "/" + view['image']

    def show_view(self, ax, view):
        try:
            crop, (x0, y0, x1, y1) = crop_around_box(Image.open(self.image_path(view)).convert('RGB'),
                                                    view['bbox'], self.args.context, self.args.display_px)
            ax.imshow(crop)
            ax.add_patch(matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='red', linewidth=1.5))
        except Exception:
            ax.text(0.5, 0.5, "missing image", ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    def style(self, axes, st, label_ax):
        """Frame color shows the labels: green = changed, blue = pickup, gray = neither."""
        color = CHANGED_COLOR if st['changed'] else (PICKUP_COLOR if st['pickup'] else PLAIN_COLOR)
        width = 4 if (st['changed'] or st['pickup']) else 1
        for ax in axes:
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(width)
        tags = ("CHANGED " if st['changed'] else "") + ("PICKUP" if st['pickup'] else "")
        label_ax.set_xlabel(tags, color=PICKUP_COLOR if st['pickup'] and not st['changed'] else CHANGED_COLOR,
                            fontsize=10, fontweight='bold')

    def style_panel(self, panel, tag, st):
        """Panel border shows the labels: thick green = changed, thick blue = pickup only, thin gray = neither."""
        color = CHANGED_COLOR if st['changed'] else (PICKUP_COLOR if st['pickup'] else PLAIN_COLOR)
        panel.set_edgecolor(color)
        panel.set_linewidth(6 if (st['changed'] or st['pickup']) else 1)
        tag.set_text(("CHANGED  " if st['changed'] else "") + ("PICKUP" if st['pickup'] else ""))
        tag.set_color('darkgreen' if st['changed'] else 'royalblue')

    # ---------------------------------------------------------- main page

    def draw_page(self):
        self.fig.clf()
        self.axes_owner = {}
        if self.expanded is not None:
            self.draw_expansion(self.expanded)
            return
        if self.page >= len(self.pages):
            self.fig.suptitle("All pages labeled - press q to quit, <- to review", fontsize=14)
            self.fig.canvas.draw_idle()
            return
        prompt, records = self.pages[self.page]
        cols, rows = self.tile_grid()
        outer = self.fig.add_gridspec(rows, cols, left=0.02, right=0.98, bottom=0.02, top=0.87, wspace=0.10, hspace=0.42)
        tile_axes = []
        for t, record in enumerate(records):
            inner = outer[t // cols, t % cols].subgridspec(1, self.args.views_per_tile, wspace=0.0)
            axes = [self.fig.add_subplot(inner[0, k]) for k in range(self.args.views_per_tile)]
            views = pick_views(record, self.args.views_per_tile, self.args.seed)
            for k, ax in enumerate(axes):
                self.axes_owner[ax] = record
                # push the views of one cluster together: left view to the right edge of its slot, etc.
                ax.set_anchor('E' if k == 0 else ('W' if k == len(axes) - 1 else 'C'))
                if k < len(views):
                    self.show_view(ax, views[k])
                else:
                    ax.axis('off')
            tile_axes.append((t, record, axes))

        # once the images are laid out, wrap each cluster's views in one colored panel (checkerboard colors)
        self.fig.canvas.draw()
        self.tiles = {}
        for t, record, axes in tile_axes:
            boxes = [ax.get_position(original=False) for ax in axes if ax.images]
            x0, y0 = min(b.x0 for b in boxes) - 0.005, min(b.y0 for b in boxes) - 0.010
            x1, y1 = max(b.x1 for b in boxes) + 0.005, max(b.y1 for b in boxes) + 0.032
            fill = PANEL_FILLS[((t // cols) + (t % cols)) % 2]
            panel = matplotlib.patches.FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.002",
                                                      transform=self.fig.transFigure, facecolor=fill, zorder=-1)
            self.fig.patches.append(panel)
            self.fig.text(x0 + 0.005, y1 - 0.024, f"cluster {record['uid'].split('/')[-1]}   ({record['n_views']} views)",
                          fontsize=10, fontweight='bold')
            tag = self.fig.text(x1 - 0.005, y1 - 0.024, "", fontsize=10, fontweight='bold', ha='right')
            self.tiles[record['uid']] = (panel, tag)
            self.style_panel(panel, tag, self.state(prompt, record['uid']))
        which = "BEFORE (render)" if self.show_before else "AFTER"
        n_done = sum(1 for p, recs in self.pages for r in recs if r['uid'] in self.labels[p])
        n_all = sum(len(recs) for _, recs in self.pages)
        self.fig.suptitle(f"{prompt}  -  page {self.page + 1}/{len(self.pages)}  -  {n_done}/{n_all} labeled  -  "
                          f"showing {which}\nleft = changed   right = pickup   middle/e = expand   "
                          f"space/-> = save & next   <- = back   t = toggle   q = quit", fontsize=11)
        self.fig.canvas.draw_idle()

    def tile_grid(self):
        """Tile columns/rows that make the square crops as large as possible in the window."""
        w, h = self.fig.get_size_inches()
        best = None
        for cols in range(1, self.args.per_page + 1):
            rows = (self.args.per_page + cols - 1) // cols
            size = min(w / (cols * self.args.views_per_tile), h / rows)
            if best is None or size > best[0]:
                best = (size, cols, rows)
        return best[1], best[2]

    def restyle(self, record):
        prompt = self.pages[self.page][0]
        if self.expanded is not None:
            self.style(self.expanded_axes, self.state(prompt, record['uid']), self.expanded_axes[0])
            self.fig.canvas.draw_idle()
            return
        tile = self.tiles.get(record['uid'])
        if tile:
            self.style_panel(tile[0], tile[1], self.state(prompt, record['uid']))
            self.fig.canvas.draw_idle()

    def toggle(self, record, button):
        prompt = self.pages[self.page][0]
        st = self.state(prompt, record['uid'])
        if button == 1:
            st['changed'] = not st['changed']
        elif button == 3:
            st['pickup'] = not st['pickup']

    def on_click(self, event):
        record = self.axes_owner.get(event.inaxes)
        if record is None:
            return
        if event.button == 2:
            self.expanded = None if self.expanded is not None else record
            self.draw_page()
            return
        self.toggle(record, event.button)
        self.restyle(record)

    def commit_page(self):
        """Save every cluster on the current page with its on-screen labels."""
        prompt, records = self.pages[self.page]
        for r in records:
            st = self.state(prompt, r['uid'])
            old = self.labels[prompt].get(r['uid'], {})
            name = old.get('name', '') if st['changed'] else "unchanged"
            if st['changed'] and name == "unchanged":
                name = ""
            self.labels[prompt][r['uid']] = {'changed': st['changed'], 'pickup': st['pickup'], 'name': name}
        save_labels(self.labels_path(prompt), self.labels[prompt])

    def on_key(self, event):
        if self.expanded is not None:             # in the expanded view only t, e/Esc and q apply
            if event.key in ('e', 'escape'):
                self.expanded = None
                self.draw_page()
            elif event.key == 't':
                self.show_before = not self.show_before
                self.draw_page()
            elif event.key == 'q':
                plt.close('all')
            return
        if event.key == 'e':
            record = self.axes_owner.get(event.inaxes)
            if record is not None:
                self.expanded = record
                self.draw_page()
        elif event.key in (' ', 'right'):
            if self.page < len(self.pages):
                self.commit_page()
                self.page += 1
                self.draw_page()
        elif event.key == 'left':
            if self.page > 0:
                self.page -= 1
                self.draw_page()
        elif event.key == 't':
            self.show_before = not self.show_before
            self.draw_page()
        elif event.key == 'q':
            plt.close('all')

    # ---------------------------------------------------------- expansion (drawn in the same window)

    def draw_expansion(self, record):
        """All of one cluster's views (up to --max_expand) in place of the page."""
        prompt = self.pages[self.page][0]
        views = pick_views(record, self.args.max_expand, self.args.seed)
        cols = min(7, len(views))
        rows = (len(views) + cols - 1) // cols
        gs = self.fig.add_gridspec(rows, cols, left=0.02, right=0.98, bottom=0.02, top=0.88, wspace=0.03, hspace=0.12)
        self.expanded_axes = []
        for i, v in enumerate(views):
            ax = self.fig.add_subplot(gs[i // cols, i % cols])
            self.show_view(ax, v)
            ax.set_title(v['frame_id'], fontsize=8)
            self.axes_owner[ax] = record
            self.expanded_axes.append(ax)
        self.style(self.expanded_axes, self.state(prompt, record['uid']), self.expanded_axes[0])
        which = "BEFORE (render)" if self.show_before else "AFTER"
        self.fig.suptitle(f"EXPANDED: {record['uid']}  -  {len(views)} of {record['n_views']} views  -  showing {which}\n"
                          f"left = changed   right = pickup   middle click / e / Esc = back to page   t = toggle", fontsize=11)
        self.fig.canvas.draw_idle()

    def run(self):
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.draw_page()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label change candidates by clicking, several clusters per page')
    parser.add_argument('config_dir', type=str, help='configuration folder, e.g. <run>/change2.0_results/sam3_0.1_openVocab')
    parser.add_argument('--prompts', type=str, nargs='*', default=[], help='prompts to label (default = every prompt in run_settings.json)')
    parser.add_argument('--per_page', type=int, default=8, help='clusters per page (default=8)')
    parser.add_argument('--views_per_tile', type=int, default=3, help='views shown per cluster on a page (default=3)')
    parser.add_argument('--max_expand', type=int, default=21, help='views shown in the expansion window (default=21)')
    parser.add_argument('--context', type=float, default=2.5, help='crop side as a multiple of the box size (default=2.5)')
    parser.add_argument('--display_px', type=int, default=256, help='crops are shrunk to at most this many pixels per side for display (default=256)')
    parser.add_argument('--seed', type=int, default=0, help='seed for choosing which views are shown (default=0)')
    args = parser.parse_args()

    config_dir = args.config_dir.rstrip('/')
    with open(config_dir + "/run_settings.json") as f:
        settings = json.load(f)
    labeler = Labeler(config_dir, settings, args.prompts if args.prompts else settings['prompts'], args)
    print(f"{sum(len(r) for _, r in labeler.pages)} clusters on {len(labeler.pages)} pages, starting at page {labeler.page + 1}")
    labeler.run()
    if labeler.no_views:
        print(f"{len(labeler.no_views)} clusters have no views and were not shown: {labeler.no_views}")
    print("labels saved")