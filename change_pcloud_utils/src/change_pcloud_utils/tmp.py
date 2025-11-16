# import cv2
# import numpy as np
# import argparse
# from PIL import Image
# import pdb

# def estimate_blur(gray_np,step=5,filter_size=50):
#     laplacian=cv2.Laplacian(gray_np, cv2.CV_64F)
#     max_row=laplacian.shape[0]-filter_size
#     max_col=laplacian.shape[1]-filter_size
#     Lpl_var=np.zeros((int(np.floor(max_row/step)),int(np.floor(max_col/step))),dtype=float)
#     for var_row,row in enumerate(range(0,laplacian.shape[0]-50,5)):
#         for var_col, col in enumerate(range(0,laplacian.shape[1]-50,5)):
#             Lpl_var[var_row,var_col]=laplacian[row:(row+filter_size),col:(col+filter_size)].var()
#     blur=cv2.resize(Lpl_var,(gray_np.shape[1],gray_np.shape[0]))
#     return blur

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('image', type=str,help='image to check for blur')
#     parser.add_argument('--threshold', type=float, default=100,help='blur threshold to use (default=100)')
#     args=parser.parse_args()

#     # colorI=Image.open(args.image,-1)
#     image=cv2.imread(args.image,-1)
#     if len(image.shape)>2 and image.shape[2]>1:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         blur=estimate_blur(gray)
#         color=image*np.tile((blur>args.threshold)[:,:,np.newaxis],(1,1,3))
#         cv2.imshow("color",color)        
#     else:
#         blur=estimate_blur(image)
#         depth=image*(blur>args.threshold)
#         cv2.imshow("depth",depth)
#     cv2.waitKey(0)
    
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

fName="/data2/datasets/living_room/specAI2/6_24_v1//images_combined//new_frame_00059.jpg"
raw_image = Image.open(fName).convert("RGB")
input_points=[[[1478,379]]]

# img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
# input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores
import cv2
import numpy as np
cv2.imshow("mask",masks[0].cpu().numpy()[0][0].astype(float))
cv2.imshow("image",np.array(raw_image))
cv2.waitKey(0)

import pdb
pdb.set_trace()





    

