# import numpy as np
# import torch

# from matplotlib import pyplot as plt
# from PIL import Image
# from transformers import AutoModel
# from torchvision.io import read_image

# device = 'cuda'

# def plot_qualitative(image, sim, palette):
#     qualitative_plot = np.zeros((sim.shape[0], sim.shape[1], 3)).astype(np.uint8)

#     for j in list(np.unique(sim)):
#         qualitative_plot[sim == j] = np.array(palette[j])
#     plt.axis('off')
#     plt.imshow(image)
#     plt.imshow(qualitative_plot, alpha=0.6)
#     plt.tight_layout()

# input_file = "/data3/datasets/smart_change/j234/changes/changes5/rotated/color_00053.png"
# image = Image.open(input_file).convert("RGB")

# model = AutoModel.from_pretrained("lorebianchi98/Talk2DINO-ViTB", trust_remote_code=True).to(device).eval()



# with torch.no_grad():
#     text_embed = model.encode_text("robot lab pictures")
#     image_embed = model.encode_image(image)

# text_embed.shape, image_embed.shape



# # normalize the features to perform cosine similarity
# text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
# image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

# similarity = (image_embed @ text_embed.T).squeeze(0, -1).cpu().numpy()

# patches_x_row = similarity.shape[0]**0.5
# plt.figure(figsize=(10, 10))
# plt.axis('off')
# plt.imshow(similarity.reshape(int(patches_x_row), int(patches_x_row)), cmap='magma')
# plt.colorbar()
# plt.tight_layout()
# plt.show()

# with_background = False
# text = "bottle, power cord, table".replace("_", " ").split(",")


# img = read_image(input_file).to(device).float().unsqueeze(0)
# palette = [
#     [255, 0, 0],
#     [255, 255, 0],
#     [0, 255, 0],
#     [0, 255, 255],
#     [0, 0, 255],
#     [128, 128, 128]
# ]
# if len(text) > len(palette):
#     for _ in range(len(text) - len(palette)):
#         palette.append([np.random.randint(0, 255) for _ in range(3)])
        
# if with_background:
#     palette.insert(0, [0, 0, 0])
#     model.with_bg_clean = True

# with torch.no_grad():
#     text_emb = model.build_dataset_class_tokens("sub_imagenet_template", text)
#     text_emb = model.build_text_embedding(text_emb)
    
#     mask, _ = model.generate_masks(img, img_metas=None, text_emb=text_emb, classnames=text, apply_pamr=True)
#     if with_background:
#         background = torch.ones_like(mask[:, :1]) * 0.55
#         mask = torch.cat([background, mask], dim=1)
    
#     mask = mask.argmax(dim=1)
    
# plot_qualitative(img.cpu()[0].permute(1,2,0).int().numpy(), mask.cpu()[0].numpy(), palette)

import torch
from transformers import AutoModel
from torchvision.io import read_image
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModel.from_pretrained("lorebianchi98/Talk2DINOv3-ViTB").to(device).eval()

# Load and preprocess image
input_file = "/data3/datasets/smart_change/j234/changes/changes5/rotated/color_00053.png"
image = Image.open(input_file).convert("RGB")
img = read_image(input_file).to(device).float().unsqueeze(0)
# image = read_image(input_file).float() / 255.0  # shape: [3, H, W]
# image = resize(image, [518, 518])  # required input size
# image = image.unsqueeze(0).to(device)  # shape: [1, 3, H, W]

import pdb
text = "bottle, power cord, table".replace("_", " ").split(",")
with_background = True

with torch.no_grad():
    text_emb = model.build_dataset_class_tokens("sub_imagenet_template", text)
    text_emb = model.build_text_embedding(text_emb)
    
    mask, _ = model.generate_masks(img, img_metas=None, text_emb=text_emb, classnames=text, apply_pamr=True)
    if with_background:
        background = torch.ones_like(mask[:, :1]) * 0.55
        mask = torch.cat([background, mask], dim=1)
    
    # mask = mask.argmax(dim=1)
    pdb.set_trace()

# with torch.no_grad():
#     text_embed = model.encode_text("general clutter")  # shape: [1, D]
#     image_embed = model.encode_image(image)           # shape: [1, N, D]

# # Normalize
# text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
# image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

# # Cosine similarity
# similarity = torch.matmul(image_embed, text_embed.T).squeeze()  # shape: [N]

# # DINOv3 ViTB has 28x28 patch grid
# mask = similarity.view(28, 28).cpu().numpy()

# # Optional: resize to original image size
# import cv2
# mask_resized = cv2.resize(mask, (518, 518), interpolation=cv2.INTER_CUBIC)

# # Threshold to get binary mask
# binary_mask = (mask_resized > 0.3).astype(float)  # adjust threshold as needed

# plt.imshow(binary_mask, cmap='gray')
# plt.title("Segmentation Mask: General Clutter")
# plt.axis('off')
# plt.show()