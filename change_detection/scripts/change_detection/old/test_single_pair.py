from accumulate_evidence import clip_seg
import argparse
from PIL import Image
import numpy as np
import cv2
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image1',type=str,help='location of nerfactor directory containing config.yml and dapaparser_transforms.json')
    parser.add_argument('image2',type=str,help='location of the transforms file to render')
    parser.add_argument('prompt',type=str,help='where to save the resulting images')
    parser.add_argument('--threshold',type=float, default=0.5, help="threshold to apply for change detection (default=0.5)")
    args = parser.parse_args()

    prompts=[args.prompt]
    CSmodel=clip_seg(prompts)
    image1 = Image.open(args.image1)
    image2 = Image.open(args.image2)
    clipV1=CSmodel.process_image(image1)
    clipV2=CSmodel.process_image(image2)
    pdb.set_trace()

    delta=clipV2[0]-clipV1[0]
    pos_mask=(delta>args.threshold).astype(np.uint8)
    neg_mask=(delta<-args.threshold).astype(np.uint8)

    image1=np.array(image1)
    image2=np.array(image2)
    res_neg=cv2.bitwise_and(image1,image1,mask=neg_mask)
    res_pos=cv2.bitwise_and(image2,image2,mask=pos_mask)

    # Missing info
    print("Missing objects")
    cv2.imshow("delta",res_neg)
    cv2.waitKey(0)
    print("Added Objects")
    cv2.imshow("delta",res_pos)
    cv2.waitKey(0)
