"""
SAM3 text-prompted segmentation using the Hugging Face Transformers implementation (facebook/sam3).

Same interface as the previous (Meta repository) wrapper:
    seg = sam3_segmentation(prompts, threshold)
    seg.process_image(pil_image)
    seg.get_prob_array(prompt)   # (H, W) score map: detection confidence where a detected mask covers the
                                 #   pixel (max over overlapping detections), 0 elsewhere
Thresholds:
    threshold        detections with confidence below this are dropped (the only tunable value)
    mask_threshold   0.5, binarization of each detection's mask; the standard SAM setting, same as the
                     Meta implementation's default
The image is encoded once per call to process_image; each prompt's text is tokenized once at startup.
"""
import pickle
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from segmentation_utils.segmentation import image_segmentation

MODEL_NAME = "facebook/sam3"


class sam3_segmentation(image_segmentation):
    def __init__(self, prompts, threshold, mask_threshold=0.5, model_name=MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sam3Model.from_pretrained(model_name).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained(model_name)

        self.prompts = prompts
        self.default_threshold = threshold
        self.mask_threshold = mask_threshold
        self.id2label = {idx: key for idx, key in enumerate(self.prompts)}
        self.label2id = {self.id2label[key]: key for key in self.id2label}
        # tokenize every prompt once; reused for every image
        self.text_inputs = {p: self.processor(text=p, return_tensors="pt").to(self.device) for p in self.prompts}
        self.clear_data()

    def process_file(self, fName, save_fileName=None):
        image = Image.open(fName)
        outputs = self.process_image(image)
        if save_fileName is not None:
            save_data = {'outputs': outputs, 'image_size': image.size, 'prompts': self.prompts}
            with open(save_fileName, 'wb') as handle:
                pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return np.array(image)

    def process_image(self, image: Image):
        """Segment every prompt in a PIL image. Returns the post-processed results of the last prompt
        (or None on failure); per-prompt results are stored for get_prob_array / get_mask."""
        self.clear_data()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        try:
            img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            target_sizes = img_inputs.get("original_sizes").tolist()
            results = None
            with torch.no_grad():
                vision_embeds = self.model.get_vision_features(pixel_values=img_inputs.pixel_values)
                for text in self.prompts:
                    outputs = self.model(vision_embeds=vision_embeds, **self.text_inputs[text])
                    results = self.processor.post_process_instance_segmentation(
                        outputs, threshold=self.default_threshold, mask_threshold=self.mask_threshold,
                        target_sizes=target_sizes)[0]
                    self.set_data(results, image.size, self.label2id[text])
        except Exception as e:
            print(f"Exception during SAM3 inference: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
        return results

    def set_data(self, results, image_size, cls):
        """Score map = max over detections of (confidence x binary mask); also masks, boxes and max score."""
        width, height = image_size
        scores = results['scores']
        if len(scores) > 0:
            masks = results['masks'].to(self.device).bool()
            scores = scores.to(self.device).float()
            self.probs[cls] = (scores[:, None, None] * masks.float()).max(dim=0).values
            self.masks[cls] = masks.any(dim=0)
            self.max_probs[cls] = scores.max().item()
            self.boxes[cls] = [(s.item(), b.floor().cpu().numpy().astype('int')) for s, b in zip(scores, results['boxes'])]
        else:
            self.probs[cls] = torch.zeros((height, width), device=self.device)
            self.masks[cls] = torch.zeros((height, width), dtype=torch.bool, device=self.device)
            self.max_probs[cls] = 0.0
            self.boxes[cls] = []


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description='Run SAM3 on one image and show the score map for one prompt')
    parser.add_argument('image', type=str, help='image to process')
    parser.add_argument('tgt_class', type=str, help='text prompt')
    parser.add_argument('--threshold', type=float, default=0.1, help='detection confidence threshold (default=0.1)')
    args = parser.parse_args()

    seg = sam3_segmentation([args.tgt_class], args.threshold)
    image = Image.open(args.image).convert('RGB')
    if seg.process_image(image) is None:
        raise SystemExit("SAM3 failed on this image")
    prob = seg.get_prob_array(args.tgt_class)
    print(f"{len(seg.boxes[0])} detections, max score {seg.max_probs[0]:.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(image)
    ax[0].set_title("image")
    ax[1].imshow(prob.cpu().numpy(), vmin=0, vmax=1)
    ax[1].set_title(f"'{args.tgt_class}' score map")
    for a in ax:
        a.axis('off')
    plt.show()