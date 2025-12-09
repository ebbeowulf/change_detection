from change_pcloud_utils.llm_query_tools import single_level_results_template, send_data_to_llm
import numpy as np
import cv2
import pdb

NUM_MULTI_IMAGES=4  # max number of images to send for processing at a given time

# First attempt at a cluster filter using an LLM
#   the multi-view-cluster-filter tries to show the LLM a bunch of images of the object
#   at the same time to remove bad candidates
class multi_view_cluster_filter():
    def __init__(self, image_scale:float=1.0):
        self.RESULTS_TEMPLATE=single_level_results_template(["object","is_pickup"],[str,bool],["<type of object>", "<True/False>"])
        #self.SUMMARY_TEMPLATE=single_level_results_template(["object"],[str],["<describe object in 5 words or less>"])
        self.TASK_DESCRIPTION = ['We are deciding on tasks for a robot that can pick small stuff up and put them away.',
                    'The robot should pick up things left behind by people that used the room.'
                    'This includes all small stuff that does not normally belong in this kind of room.'
                    'The robot should not pickup decorations or expensive electronics.'
                    'The object surrounded by the blue box in the provided image(s) has been identified by the robot as a candidate for picking up.',
                    'Is the object surrounded by the blue box in these images an object that the robot should pick up and put away?']
        
        self.PROMPT=""
        for task in self.TASK_DESCRIPTION:
            self.PROMPT+=task
        self.PROMPT+="Return an answer in JSON format as " + self.RESULTS_TEMPLATE.generate_format_prompt()
        self.scale=image_scale
        print(self.PROMPT)

    def evaluate_cluster(self, all_images:list):
        def get_images_from_set(all_images:list, key_set, scale:float=1.0):                     
            return [ cv2.resize(all_images[key]['new'], (0,0), fx=scale, fy=scale) for key in key_set] 

        keys_with_new = np.array([key for key, subdict in all_images.items() if "new" in subdict])
        all_results=[]
        cnt_positive=0
        cnt_negative=0
        if keys_with_new.shape[0]>0:
            if keys_with_new.shape[0]<=NUM_MULTI_IMAGES:
                text_results= send_data_to_llm(self.PROMPT, get_images_from_set(all_images, keys_with_new.tolist(), self.scale))
                res=self.RESULTS_TEMPLATE.recover_json(text_results)
                if res['is_pickup']:
                    cnt_positive+=1
                else:
                    cnt_negative+=1
                all_results.append(res)
            else:
                num_runs=int(np.ceil(keys_with_new.shape[0]/NUM_MULTI_IMAGES))
                for i in range(num_runs):
                    selected_images = np.random.choice(keys_with_new, NUM_MULTI_IMAGES, replace=False)
                    text_results= send_data_to_llm(self.PROMPT, get_images_from_set(all_images, selected_images.tolist(), self.scale))
                    res=self.RESULTS_TEMPLATE.recover_json(text_results)
                    if res['is_pickup']:
                        cnt_positive+=1
                    else:
                        cnt_negative+=1
                    all_results.append(res)
        
        print(all_results)
        if cnt_positive==0:
            return 0
        return (cnt_positive / (cnt_positive+cnt_negative))
    
# First attempt at a cluster filter using an LLM
#   the multi-view-cluster-filter tries to show the LLM a bunch of images of the object
#   at the same time to remove bad candidates
class before_and_after_cluster_filter():
    def __init__(self, description:list=None):
        self.RESULTS_TEMPLATE=single_level_results_template(["new_object","is_pickup"],
                                                            [bool,bool],
                                                            ["<True/False>", "<True/False>"])
        if description is None:
            self.TASK_DESCRIPTION = ['The attached image includes two images of the same location captured at different times.',
                                    'Images might be a little blurry due to either reconstruction error or motion blur.'
                                    'A blue box is drawn around the same area in both images.',
                                    'Two questions: (1) Is there a new object present in the right-most image?',
                                    '(2) Is the new object inside the blue box small enough to be picked up by a robot with a single two fingered gripper with a 3 kg weight limit?']            
        else:
            self.TASK_DESCRIPTION = description

        self.PROMPT=""
        for task in self.TASK_DESCRIPTION:
            self.PROMPT+=task
        self.PROMPT+="Return an answer in JSON format as " + self.RESULTS_TEMPLATE.generate_format_prompt()
        print(self.PROMPT)

    def merge_images_with_strip(self, img1: np.ndarray, img2: np.ndarray, strip_width: int = 20) -> np.ndarray:
        """
        Merge two images of the same shape side-by-side with a white vertical strip in between.

        Parameters:
            img1 (np.ndarray): First image array.
            img2 (np.ndarray): Second image array.
            strip_width (int): Width of the white strip between images.

        Returns:
            np.ndarray: Merged image.
        """
        assert img1.shape == img2.shape, "Images must have the same shape"
        assert img1.ndim == 3 and img1.shape[2] == 3, "Images must be color (H, W, 3)"

        height, width, channels = img1.shape
        white_strip = np.ones((height, strip_width, channels), dtype=np.uint8) * 255

        merged = np.concatenate((img1, white_strip, img2), axis=1)
        return merged

    def evaluate_image_pair(self, before_image:np.ndarray, after_image:np.ndarray):
        merged_image=self.merge_images_with_strip(before_image, after_image)
        text_results=send_data_to_llm(self.PROMPT, [merged_image])
        return self.RESULTS_TEMPLATE.recover_json(text_results)

    def evaluate_multiple_image_pairs(self, 
                                      all_images, # format of [key]{'new': np.ndarray, 'render': np.ndarray}, generated by draw_boxes_around_cluster function (pcloud_cluster_utils.py)
                                      after_key='new', # after image stored under what name in the all_images dict?
                                      before_key='render' # before image stored under what name in the all_images dict?
                                ):
        all_results={}
        cnt_total=0
        cnt_new=0
        cnt_is_pickup=0
        for key in all_images:
            if after_key in all_images[key] and before_key in all_images[key]:
                cnt_total+=1
                all_results[key]=self.evaluate_image_pair(all_images[key][before_key],all_images[key][after_key])
                if 'new_object' in all_results[key] and all_results[key]['new_object']:
                    cnt_new+=1
                if 'is_pickup' in all_results[key] and all_results[key]['is_pickup']:
                    cnt_is_pickup+=1
        
        if cnt_total>0:
            return cnt_is_pickup/cnt_total, cnt_new/cnt_total
        else:
            return 0, 0
        # Majority vote - do either pickup or new fail a majority vote?
        # if cnt_is_pickup<0.5*(cnt_total) or cnt_new<0.5*(cnt_total):
        #     return False
        # return True
