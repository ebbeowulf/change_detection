import torch
from PIL import Image
import open_clip
import argparse
import pdb
from image_set import read_image_csv

labels=["chair", "printer", "plant", "tv", "monitor", "paper", "binder", "keys", "food", "painting", 
        "poster", "book", "keyboard", "laptop", "robot", "cart", "flag", "fire extinguisher", 
        "desk", "table", "phone", "clothing", "backpack", "cabinet"]

class process_images():
    def __init__(self, labels):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.labels=labels
        self.text=self.tokenizer(self.labels)

    def score(self, image_features, text_features, apply_softmax=True):
        if apply_softmax:
            return (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return (100.0 * image_features @ text_features.T)

    def process_image(self, image:Image):
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            return image_features, text_features

    def read_and_process_image(self, image_loc:str, use_softmax=False):
        image = self.preprocess(Image.open(image_loc)).unsqueeze(0)
        image_features, text_features=self.process_image(image)
        res = self.score(image_features, text_features, use_softmax)
        return res.numpy()[0].tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images_csv',type=str,help='location of image csv file to process')
    parser.add_argument('image_dir',type=str,help='location of images in file system')
    parser.add_argument('output_file',type=str,help='location of output file to save')
    parser.add_argument('--softmax', action='store_true')
    parser.add_argument('--no-softmax', dest='softmax', action='store_false')
    parser.set_defaults(softmax=True)    
    args = parser.parse_args()

    PI=process_images(labels)
    all_images=read_image_csv(args.images_csv)

    with open(args.output_file,'w') as fout:
        print("image_name ", file=fout, end='')
        for lbl in labels:
            print(', %s'%(lbl),file=fout, end='')
        print("",file=fout)

        # Now - for each image get the probability and 
        for im in all_images:
            res=PI.read_and_process_image(args.image_dir+'/'+im['name'],args.softmax)
            print(res)

            print(im['name'], file=fout, end='')
            for val in res:
                print(', %f'%(val),file=fout, end='')
            print("",file=fout)



