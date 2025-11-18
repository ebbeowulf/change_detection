import argparse
from PIL import Image
import numpy as np
import cv2
import pdb

def build_dbscan_boxes(prob_image, mask, eps=10, min_samples=20, MAX_CLUSTERING_SAMPLES=50000):
    from sklearn.cluster import DBSCAN

    rows,cols=np.nonzero(mask)
    xy_grid_pts=np.vstack((cols,rows)).transpose()
    scores=prob_image[rows,cols]
    if xy_grid_pts is None or xy_grid_pts.shape[0]<min_samples:
        return []

    # Need to constrain the maximum number of points - else dbscan will be extremely slow
    if xy_grid_pts.shape[0]>MAX_CLUSTERING_SAMPLES:        
        rr=np.random.choice(np.arange(xy_grid_pts.shape[0]),size=MAX_CLUSTERING_SAMPLES)
        xy_grid_pts=xy_grid_pts[rr]
        scores=scores[rr]

    CL2=DBSCAN(eps=eps, min_samples=min_samples).fit(xy_grid_pts,sample_weight=scores)
    boxes=[]
    for idx in range(10):
        whichP=np.where(CL2.labels_== idx)            
        if len(whichP[0])<1:
            break
        box=np.hstack((xy_grid_pts[whichP].min(0),xy_grid_pts[whichP].max(0)))
        boxes.append((scores[whichP].max(),box))
    return boxes

def is_box_overlap(bbox1,bbox2):
    from shapely import Polygon
    polygon1 = Polygon([bbox1[:2], [bbox1[2],bbox1[1]], bbox1[2:],[bbox1[0],bbox1[3]]])
    polygon2 = Polygon([bbox2[:2], [bbox2[2],bbox2[1]], bbox2[2:],[bbox2[0],bbox2[3]]])
    return polygon1.intersects(polygon2)

def box_iou(bbox1,bbox2):
    from shapely import Polygon
    polygon1 = Polygon([bbox1[:2], [bbox1[2],bbox1[1]], bbox1[2:],[bbox1[0],bbox1[3]]])
    polygon2 = Polygon([bbox2[:2], [bbox2[2],bbox2[1]], bbox2[2:],[bbox2[0],bbox2[3]]])
    if polygon1.intersects(polygon2):
        p_area1=polygon1.area
        p_area2=polygon2.area
        i_area=polygon1.intersection(polygon2).area
        return i_area/(p_area1+p_area2-i_area)
    return 0.0

def sam_segmentation(colorI, xy_points:list):
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"    
    from transformers import SamModel, SamProcessor

    sam_model=SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE)
    sam_processor=SamProcessor.from_pretrained("facebook/sam-vit-huge")
    inputs = sam_processor(colorI, input_points=[xy_points], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks=sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    return masks[0].data[0][0].cpu().numpy()
    
def refine_boxes_with_sam(colorI,
                          pos_mask,
                          db_boxes):
    if len(db_boxes)==0:
        return [],np.zeros((colorI.height,colorI.width),dtype=bool)

    mask_combo=None
    sam_boxes=[]
    for box in db_boxes:
        box_dim=np.array(box[1][2:])-np.array(box[1][:2])
        deltaDim=(box_dim/20).astype(int)
        new_box=np.hstack((np.array(box[1][:2])+deltaDim,np.array(box[1][2:])-deltaDim))
        subR=pos_mask[new_box[1]:new_box[3],new_box[0]:new_box[2]]
        rowD,colD=np.nonzero(subR)
        whichP=np.random.choice(np.arange(rowD.shape[0]),10)
        xy_points=np.vstack((colD[whichP]+box[1][0],rowD[whichP]+box[1][1])).transpose().tolist()
        sam_mask=sam_segmentation(colorI,xy_points)        
        # ctr=((box[1][:2]+box[1][2:])/2.0).astype(int)
        # sam_mask=sam_segmentation(colorI,[ctr.tolist()])        
        if mask_combo is None:
            mask_combo=sam_mask
        else:
            mask_combo*=sam_mask
        rowS,colS=np.nonzero(sam_mask)
        sbox=[colS.min(),rowS.min(),colS.max(),rowS.max()]
        if box_iou(box[1],sbox)>0.3:
            max_prob=0
            for cbox in db_boxes:
                if is_box_overlap(sbox,cbox[1]):
                    max_prob=max((max_prob,cbox[0]))
            sam_boxes.append((max_prob,sbox))
    
    # Now merge overlapping boxes
    count_boxes=len(sam_boxes)+1
    while len(sam_boxes)>1 and count_boxes>len(sam_boxes):
        count_boxes=len(sam_boxes)
        old_boxes=sam_boxes
        sam_boxes=[]
        is_available=np.ones((len(old_boxes)),dtype=bool)
        for box_idx, box in enumerate(old_boxes):
            if not is_available[box_idx]:
                continue
            is_available[box_idx]=False
            tgt_box=box
            for box_idx2 in range(box_idx+1,len(old_boxes)):
                if is_box_overlap(tgt_box[1],old_boxes[box_idx2][1]):
                    is_available[box_idx2]=False
                    combo=np.vstack((tgt_box[1],old_boxes[box_idx2][1]))
                    tgt_box=(max(tgt_box[0],old_boxes[box_idx2][0]),
                          np.hstack((combo[:,:2].min(0),combo[:,2:].max(0))))
            sam_boxes.append(tgt_box)
    return sam_boxes, mask_combo
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image1',type=str,help='location of before image')
    parser.add_argument('image2',type=str,help='location of the after image')
    parser.add_argument('prompt',type=str,help='where to save the resulting images')
    parser.add_argument('--threshold',type=float, default=0.3, help="fixed threshold to apply for change detection (default=0.3)")
    parser.add_argument('--draw-boxes', default=None, help="Optional draw boxes ('dbscan','sam','all')")
    parser.add_argument('--model', default='clipseg', help="Select different open vocabulary segmentation models ('clipseg','yolo-world','dino')")
    args = parser.parse_args()

    prompts=[args.prompt]
    if args.model=='clipseg':
        from segmentation_utils.clip_segmentation import clip_seg
        CSmodel=clip_seg(prompts)
    elif args.model=='yolo-world':
        from segmentation_utils.yolo_world_segmentation import yolo_world_segmentation
        CSmodel=yolo_world_segmentation(prompts)
    elif args.model=='dino':
        from segmentation_utils.dino_segmentation import dino_segmentation
        CSmodel=dino_segmentation(prompts)

    image1=Image.open(args.image1)
    CSmodel.process_image(image1)
    prob1 = CSmodel.get_prob_array(0).to('cpu').numpy()
    image2=Image.open(args.image2)
    CSmodel.process_image(image2)
    prob2 = CSmodel.get_prob_array(0).to('cpu').numpy()


    delta=(prob2-prob1)
    print(f"MAX DELTA={delta.max()}")

    mask=(delta>args.threshold)
    im_out=np.array(image2)

    if args.draw_boxes is None:
        im_out[:,:,0][mask]=255
    else:        
        dbscan_boxes=build_dbscan_boxes(delta,(delta>args.threshold).astype(np.uint8))
        if args.draw_boxes=='dbscan' or args.draw_boxes=='all':
            im_out[:,:,0][mask]=255
            for bbox in dbscan_boxes:
                im_out=cv2.rectangle(im_out, bbox[1][:2], bbox[1][2:], (255,0,0), 2)
        if args.draw_boxes=='sam' or args.draw_boxes=='all':
            sboxes, mask_combo=refine_boxes_with_sam(image2,mask,dbscan_boxes)
            im_out[:,:,2][mask_combo]=255
            for bbox in sboxes:
                im_out=cv2.rectangle(im_out, bbox[1][:2], bbox[1][2:], (0,0,255), 2)

    im_out[:,:,:3]=im_out[:,:,[2,1,0]]
    if im_out.shape[1]>480:
        import cv2
        dim=(int(im_out.shape[1]/2),int(im_out.shape[0]/2))
        im_out=cv2.resize(im_out,dim)        

    print("Added Objects")
    cv2.imshow("delta",im_out)
    cv2.waitKey(0)
