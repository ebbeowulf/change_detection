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
    image1 = CSmodel.process_file(args.image1)[:,:,[2,1,0]]
    prob1 = CSmodel.get_prob_array(0).to('cpu').numpy()
    image2 = CSmodel.process_file(args.image2)[:,:,[2,1,0]]
    prob2 = CSmodel.get_prob_array(0).to('cpu').numpy()

    delta=(prob2-prob1)
    print(f"MAX DELTA={delta.max()}")
    
    if args.threshold is not None:     
        mask=(delta>args.threshold).astype(np.uint8)
    elif args.pct_threshold is not None:
        thresh=delta.max()*(args.pct_threshold)
        mask=(delta>thresh).astype(np.uint8)
    else:
        print("Must use either threshold or fixed threshold")
        import sys
        sys.exit(-1)

    d_mask = (1-mask).astype(np.uint8)
    red_image=np.ones(image2.shape,dtype=np.uint8)
    red_image[:,:,2]=255
    pos=cv2.bitwise_and(image2,image2,mask=d_mask)+cv2.bitwise_and(red_image,red_image,mask=mask)

    if pos.shape[1]>480:
        import cv2
        dim=(int(pos.shape[1]/2),int(pos.shape[0]/2))
        pos=cv2.resize(pos,dim)        

    print("Added Objects")
    cv2.imshow("delta",pos)
    cv2.waitKey(0)
