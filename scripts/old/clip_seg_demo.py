from PIL import Image
import pdb
import cv2
import numpy as np

#from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/CLIPSeg/Zero_shot_image_segmentation_with_CLIPSeg.ipynb

#fName="/home/emartinso/data/office/office_change/changed/rotated/rgb_00070.png"
#fName="/home/emartinso/data/office/office_change/changed/rotated/rgb_00070.png"
# fName="/home/emartinso/data/office/no_person/monitor/rotated/rgb_00170.png"
# fName="/home/emartinso/data/office/no_person/bookshelf/rotated/rgb_00350.png"
fName="/home/emartinso/projects/nerfstudio/renders/no_person_monitor/rgb_00170.png"
image = Image.open(fName)
# image = cv2.imread(fName)

print("Reading model")
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# prompts=["scattered papers", "a suspicious object", "a monitor", "something broken"]
prompts=["a computer",
        "something is missing",
         "a suspicious object", 
         "signs of a break-in",
         "a package", 
         "wooden office furniture"]
inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

import torch

print("Inference")
# predict
with torch.no_grad():
  outputs = model(**inputs)

preds = outputs.logits.unsqueeze(1)
P_resized=np.zeros((preds.shape[0],image.size[1],image.size[0]),dtype=float)
for dim in range(preds.shape[0])    :
  print("%s = %f"%(prompts[dim],preds[dim,:,:,:].max()))
  P_resized[dim,:,:]=cv2.resize(preds[dim,0,:,:].numpy(),(image.size[0],image.size[1]))

print("Plotting")
import matplotlib.pyplot as plt
# visualize prediction
# pdb.set_trace()
_, ax = plt.subplots(2, 5, figsize=(15, 4))
[a.axis('off') for a in ax.flatten()]
ax[0,0].imshow(image)
half_len=int(len(prompts)/2)
# [ax[0,i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(half_len)]
# [ax[0,i+1].text(0, -15, prompts[i]) for i in range(half_len)]
# [ax[1,i-half_len].imshow(torch.sigmoid(preds[i][0])) for i in range(half_len,len(prompts))]
# [ax[1,i-half_len].text(0, -15, prompts[i]) for i in range(half_len,len(prompts))]
[ax[0,i+1].imshow((1.0/(1+np.exp(-P_resized[i])))) for i in range(half_len)]
[ax[0,i+1].text(0, -15, prompts[i]) for i in range(half_len)]
[ax[1,i-half_len].imshow((1.0/(1+np.exp(-P_resized[i])))) for i in range(half_len,len(prompts))]
[ax[1,i-half_len].text(0, -15, prompts[i]) for i in range(half_len,len(prompts))]

plt.savefig("mygraph.png")
