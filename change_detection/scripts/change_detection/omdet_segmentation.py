from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import SAM
import torch
import cv2
import numpy as np
import argparse
from change_detection.segmentation import image_segmentation
from PIL import Image
import pdb
import pickle

#from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/CLIPSeg/Zero_shot_image_segmentation_with_CLIPSeg.ipynb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class omdet_seg(image_segmentation):
    def __init__(self, prompts):
        print("Reading model")
        # Increased timeout for model download if needed
        # from transformers.file_utils import hf_hub_download
        # model_path = hf_hub_download(repo_id="omlab/omdet-turbo-swin-tiny-hf", filename="pytorch_model.bin", cache_dir=None, force_filename=None, legacy_cache_layout=False)
        # config_path = hf_hub_download(repo_id="omlab/omdet-turbo-swin-tiny-hf", filename="config.json", cache_dir=None, force_filename=None, legacy_cache_layout=False)
        # processor_path = hf_hub_download(repo_id="omlab/omdet-turbo-swin-tiny-hf", filename="preprocessor_config.json", cache_dir=None, force_filename=None, legacy_cache_layout=False)

        self.processor = AutoProcessor.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("omlab/omdet-turbo-swin-tiny-hf")
        self.sam_model = SAM('sam2.1_l.pt')
        if DEVICE==torch.device("cuda"):
            self.model.cuda()
            # Consider moving SAM model to GPU as well if memory allows
            self.sam_model.to(DEVICE)

        self.prompts=prompts
        self.id2label={idx: key for idx,key in enumerate(self.prompts)}
        self.label2id={self.id2label[key]: key for key in self.id2label }
        self.image_size = None # Store image size when data is set
        self.clear_data()

    def clear_data(self):
        # Clear stored results for a new image
        self.masks = {i: [] for i in range(len(self.prompts))}
        self.boxes = {i: [] for i in range(len(self.prompts))}
        self.scores = {i: [] for i in range(len(self.prompts))}
        self.image_size = None

    def sigmoid(self, arr):
        return (1.0/(1.0+np.exp(-arr)))

    # Modified load_file: No threshold needed, loads masks/boxes/scores
    def load_file(self, fileName):
        try:
            # Otherwise load the file
            with open(fileName, 'rb') as handle:
                save_data=pickle.load(handle)
                if save_data['prompts']==self.prompts:
                    self.masks = save_data['masks']
                    self.boxes = save_data['boxes']
                    self.scores = save_data['scores']
                    self.image_size = save_data['image_size']
                    print(f"Loaded processed data from {fileName}")
                    return True
                else:
                    print("Prompts in saved file do not match ... skipping load")
        except FileNotFoundError:
            print(f"File not found: {fileName}")
        except Exception as e:
            print(f"Error loading file {fileName}: {e}")
        # Ensure data is cleared if loading fails or prompts don't match
        self.clear_data()
        return False

    # Modified process_file: Calls new process_image, saves new structure
    def process_file(self, fName, save_fileName=None):
        try:
            image = Image.open(fName).convert("RGB") # Ensure RGB
        except Exception as e:
            print(f"Error opening image {fName}: {e}")
            return None

        # Process the image (OMDET + SAM)
        success = self.process_image(image)

        if success and save_fileName is not None:
            save_data={'masks': self.masks,
                       'boxes': self.boxes,
                       'scores': self.scores,
                       'image_size': self.image_size,
                       'prompts': self.prompts}
            try:
                with open(save_fileName, 'wb') as handle:
                    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved processed data to {save_fileName}")
            except Exception as e:
                print(f"Error saving data to {save_fileName}: {e}")

        # Convert the PIL image to opencv format and return
        return np.array(image)[:,:,::-1] # Convert RGB to BGR for OpenCV

    # Heavily modified process_image: Runs OMDET, filters, runs SAM, calls set_data
    def process_image(self, image):
        self.clear_data()
        self.image_size = image.size
        cv_image = np.array(image) # Keep in RGB for SAM

        OMDET_CONF_THRESHOLD = 0.1 # Threshold for post-processing
        NMS_THRESHOLD = 0.5 # Non-Maximum Suppression threshold

        print("Running OMDET Inference...")
        try:
            inputs = self.processor(text=self.prompts, images=image, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use processor's post-processing
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                text_labels=self.prompts, # Pass the original prompts
                target_sizes=[self.image_size[::-1]], # Target size (height, width)
                threshold=OMDET_CONF_THRESHOLD,
                nms_threshold=NMS_THRESHOLD,
            )[0] # Assuming batch size 1

            # Extract results from the dictionary
            filtered_scores = results["scores"].cpu().numpy()
            string_labels = results["classes"] # These are string labels from self.prompts
            # Convert string labels back to indices for consistency with set_data
            filtered_labels = np.array([self.label2id[lbl] for lbl in string_labels])
            # Boxes from post-processing are typically xyxy absolute
            filtered_boxes_xyxy = results["boxes"].cpu().numpy()

            print(f"OMDET post-processing found {len(filtered_boxes_xyxy)} boxes above threshold {OMDET_CONF_THRESHOLD}")

        except Exception as e:
            print(f"Exception during OMDET inference or post-processing: {e}")
            import traceback
            traceback.print_exc()
            return False

        if len(filtered_boxes_xyxy) == 0:
            print("No objects detected by OMDET above threshold, skipping SAM.")
            self.set_data(None, [], [], [], image.size)
            return True

        print("Running SAM Inference...")
        try:
            if self.sam_model is None:
                print("Error: SAM model not loaded.")
                return False

            # Boxes from post-processing are xyxy absolute
            sam_boxes_xyxy = filtered_boxes_xyxy.astype(int)
            # Clip boxes to image bounds
            h, w = cv_image.shape[:2]
            sam_boxes_xyxy[:, [0, 2]] = sam_boxes_xyxy[:, [0, 2]].clip(0, w)
            sam_boxes_xyxy[:, [1, 3]] = sam_boxes_xyxy[:, [1, 3]].clip(0, h)

            # Run SAM
            # Suppress SAM output if desired (like in example)
            # with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sam_results = self.sam_model(cv_image, bboxes=sam_boxes_xyxy)

            if not isinstance(sam_results, list):
                sam_results = [sam_results]

            print(f"SAM generated masks for {len(sam_results[0].masks.data) if sam_results and sam_results[0].masks is not None else 0} boxes")

            # Store results using the new set_data method
            # NOTE: set_data currently expects OMDET boxes in cxcywh_norm format.
            # We need to either:
            #   1. Convert filtered_boxes_xyxy back to cxcywh_norm before passing.
            #   2. Modify set_data to accept xyxy_abs format.
            # Let's choose option 1 for now to minimize changes to set_data.

            # Convert xyxy_abs back to cxcywh_norm for set_data
            if len(filtered_boxes_xyxy) > 0:
                xyxy_tensor = torch.tensor(filtered_boxes_xyxy)
                cxcywh_norm_tensor = torch.zeros_like(xyxy_tensor)
                cxcywh_norm_tensor[:, 0] = ((xyxy_tensor[:, 0] + xyxy_tensor[:, 2]) / 2) / w # cx
                cxcywh_norm_tensor[:, 1] = ((xyxy_tensor[:, 1] + xyxy_tensor[:, 3]) / 2) / h # cy
                cxcywh_norm_tensor[:, 2] = (xyxy_tensor[:, 2] - xyxy_tensor[:, 0]) / w # w
                cxcywh_norm_tensor[:, 3] = (xyxy_tensor[:, 3] - xyxy_tensor[:, 1]) / h # h
                filtered_boxes_cxcywh_norm = cxcywh_norm_tensor.numpy()
            else:
                filtered_boxes_cxcywh_norm = np.zeros((0, 4)) # Empty array if no boxes

            self.set_data(sam_results, filtered_labels, filtered_scores, filtered_boxes_cxcywh_norm, image.size)

        except Exception as e:
            print(f"Exception during SAM inference or data setting: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
            # Ensure data is cleared on SAM error
            self.clear_data()
            return False

        return True

    # New set_data: Stores SAM masks and OMDET info indexed by original prompt index
    def set_data(self, sam_results, omdet_labels, omdet_scores, omdet_boxes_cxcywh_norm, image_size):
        self.clear_data() # Start fresh
        self.image_size = image_size

        if sam_results is None or sam_results[0].masks is None or len(sam_results[0].masks.data) == 0:
            print("No SAM masks generated or provided to set_data.")
            return

        # sam_results[0].masks.data should be a tensor of shape [N, H, W]
        # where N is the number of boxes/masks
        sam_masks_tensor = sam_results[0].masks.data.cpu() # Move masks to CPU

        # Ensure N matches the length of filtered labels/scores/boxes
        num_masks = sam_masks_tensor.shape[0]
        num_omdet_results = len(omdet_labels)

        if num_masks != num_omdet_results:
            print(f"Warning: Mismatch between number of SAM masks ({num_masks}) and OMDET results ({num_omdet_results}). Alignment might be incorrect.")
            # Attempt to proceed, but this indicates a potential issue upstream
            min_len = min(num_masks, num_omdet_results)
            sam_masks_tensor = sam_masks_tensor[:min_len]
            omdet_labels = omdet_labels[:min_len]
            omdet_scores = omdet_scores[:min_len]
            omdet_boxes_cxcywh_norm = omdet_boxes_cxcywh_norm[:min_len]

        # Populate self.masks, self.boxes, self.scores using the prompt index from omdet_labels
        for i in range(len(omdet_labels)):
            prompt_index = omdet_labels[i]
            if 0 <= prompt_index < len(self.prompts):
                # Append the boolean mask, box, and score
                self.masks[prompt_index].append(sam_masks_tensor[i].numpy()) # Store mask as numpy array
                self.boxes[prompt_index].append(omdet_boxes_cxcywh_norm[i])
                self.scores[prompt_index].append(omdet_scores[i])
            else:
                print(f"Warning: OMDET label index {prompt_index} out of bounds for prompts.")

    # Modified get_mask: Returns a combined boolean mask for a prompt index
    def get_mask(self, prompt_index_or_name):
        if isinstance(prompt_index_or_name, str):
            if prompt_index_or_name not in self.label2id:
                print(f"Prompt name '{prompt_index_or_name}' not found.")
                return None
            prompt_index = self.label2id[prompt_index_or_name]
        else:
            prompt_index = prompt_index_or_name

        if not (0 <= prompt_index < len(self.prompts)):
            print(f"Prompt index {prompt_index} out of bounds.")
            return None

        if not self.masks[prompt_index]:
            # print(f"No masks found for prompt index {prompt_index} ('{self.id2label[prompt_index]}')")
            return None # No masks for this prompt

        # Combine all masks for this prompt index using logical OR
        combined_mask = np.logical_or.reduce(self.masks[prompt_index])
        return combined_mask

    # Example: Get all boxes for a specific prompt
    def get_boxes(self, prompt_index_or_name):
        if isinstance(prompt_index_or_name, str):
            if prompt_index_or_name not in self.label2id:
                return []
            prompt_index = self.label2id[prompt_index_or_name]
        else:
            prompt_index = prompt_index_or_name

        if not (0 <= prompt_index < len(self.prompts)):
             return []

        return self.boxes[prompt_index]

    # Example: Get all scores for a specific prompt
    def get_scores(self, prompt_index_or_name):
        if isinstance(prompt_index_or_name, str):
            if prompt_index_or_name not in self.label2id:
                return []
            prompt_index = self.label2id[prompt_index_or_name]
        else:
            prompt_index = prompt_index_or_name

        if not (0 <= prompt_index < len(self.prompts)):
             return []

        return self.scores[prompt_index]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=str,help='location of image to process')
    parser.add_argument('tgt_prompt',type=str, nargs='+', default=None,help='specific prompt(s) for clip class') # Allow multiple prompts
    # parser.add_argument('--threshold',type=float,default=0.2,help='(optional) threshold to apply during computation ') # Threshold now internal to OMDET filtering
    args = parser.parse_args()

    if args.tgt_prompt is None:
        print("Error: Please provide at least one target prompt.")
        exit()

    # Use omdet_seg, not clip_seg
    OS=omdet_seg(args.tgt_prompt)

    # Process the file (no threshold needed here)
    image=OS.process_file(args.image)

    if image is None:
        print("Failed to process image.")
        exit()

    # Display mask for the first prompt
    tgt_display_prompt = args.tgt_prompt[0]
    mask=OS.get_mask(tgt_display_prompt) # Use get_mask with the prompt name

    if mask is None:
        print(f"Something went wrong or no mask found for '{tgt_display_prompt}'")
    else:
        print(f"Displaying mask for '{tgt_display_prompt}'")
        # cv_image=np.array(image).astype(np.uint8)
        cv_image = image # process_file now returns BGR numpy array
        # Ensure mask is uint8 for bitwise_and
        IM=cv2.bitwise_and(cv_image,cv_image,mask=mask.astype(np.uint8))
        cv2.imshow(f"Result for {tgt_display_prompt}",IM)

        # Optionally display masks for other prompts
        if len(args.tgt_prompt) > 1:
            for other_prompt in args.tgt_prompt[1:]:
                other_mask = OS.get_mask(other_prompt)
                if other_mask is not None:
                    IM_other = cv2.bitwise_and(cv_image,cv_image,mask=other_mask.astype(np.uint8))
                    cv2.imshow(f"Result for {other_prompt}", IM_other)
                else:
                     print(f"No mask found for '{other_prompt}'")

        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
