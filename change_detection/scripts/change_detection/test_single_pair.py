from clip_segmentation import clip_seg
import argparse
from PIL import Image
import numpy as np
import cv2
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image1',type=str,help='location of before image')
    parser.add_argument('image2',type=str,help='location of the after image')
    parser.add_argument('prompt',type=str,help='where to save the resulting images')
    parser.add_argument('--threshold',type=float, default=None, help="fixed threshold to apply for change detection. Must use either threshold or pct_threshold)")
    parser.add_argument('--pct_threshold',type=float, default=None, help="Pct of max threshold to apply for change detection.")
    args = parser.parse_args()

    prompts=[args.prompt]
    CSmodel=clip_seg(prompts)
    image1 = CSmodel.process_file(args.image1)
    prob1 = CSmodel.get_prob_array(0)
    image2 = CSmodel.process_file(args.image2)
    prob2 = CSmodel.get_prob_array(0)
    # image1 = Image.open(args.image1)
    # image2 = Image.open(args.image2)
    # clipV1=CSmodel.process_image(image1)
    # clipV2=CSmodel.process_image(image2)

    delta=(prob2-prob1)

    if args.threshold is not None:     
        mask=(delta>args.threshold).astype(np.uint8)
    elif args.pct_threshold is not None:
        thresh=delta.max()*(args.pct_threshold)
        mask=(delta>thresh).astype(np.uint8)
    else:
        print("Must use either threshold or fixed threshold")
        import sys
        sys.exit(-1)

    # neg_mask=((prob1-prob2)>args.threshold).astype(np.uint8)

    # image1=np.array(image1)
    # image2=np.array(image2)
    # pdb.set_trace()
    # res_neg=cv2.bitwise_and(image1,image1,mask=neg_mask)
    # pos=cv2.bitwise_and(image2,image2,mask=mask)

    d_mask = (1-mask).astype(np.uint8)
    red_image=np.ones(image2.shape,dtype=np.uint8)
    red_image[:,:,2]=255
    pos=cv2.bitwise_and(image2,image2,mask=d_mask)+cv2.bitwise_and(red_image,red_image,mask=mask)

    pdb.set_trace()
    if pos.shape[1]>480:
        import cv2
        dim=(int(pos.shape[1]/2),int(pos.shape[0]/2))
        pos=cv2.resize(pos,dim)        


    # Missing info
    # print("Missing objects")
    # cv2.imshow("delta",res_neg)
    # cv2.waitKey(0)
    print("Added Objects")
    cv2.imshow("delta",pos)
    cv2.waitKey(0)
