import open3d as o3d
import numpy as np
import argparse
import cv2
from map_from_scannet import build_file_structure, load_camera_info
import pdb
import os
import copy
import json
from camera_params import camera_params
from rgbd_file_list import rgbd_file_list

MAX_POINTS=200000

# Global Variables
mouse_clicks=[]
image_interface=None

def mouse_cb(event,x,y,flags,param):
    global mouse_clicks, image_interface
    if event == cv2.EVENT_LBUTTONDOWN:
        print("left button down")
        if len(mouse_clicks)<2:
            mouse_clicks.append((x,y))            
        else:
            print("Cannot add more points - box is already fully defined")
        
        if len(mouse_clicks)==2:
            img=copy.copy(image_interface.fg_image)
            img=cv2.rectangle(img,mouse_clicks[0],mouse_clicks[1],color=(255,0,0))
            cv2.imshow(image_interface.window_name,img)
            cv2.waitKey(1)
    elif event == cv2.EVENT_MBUTTONDOWN:
        print("open images")

class drawn_image():
    def __init__(self, pointcloud, width=1280, height=720, window_name="environment"):
        self.image=np.zeros((height,width,3),dtype=np.uint8)
        self.height=height
        self.width=width

        pct=MAX_POINTS / len(pointcloud.points)
        if pct<1.0:
            sampled=pointcloud.random_down_sample(pct)
            self.xyz=np.array(sampled.points)
            self.rgb=np.array(sampled.colors)
        else:
            self.xyz=np.array(pointcloud.points)
            self.rgb=np.array(pointcloud.colors)

        # Have to clear NaN pixels - open3d doesn't do it for us
        validP=np.where(np.isnan(self.xyz).sum(1)==0)
        self.xyz=self.xyz[validP]
        self.rgb=self.rgb[validP]

        self.minXYZ=self.xyz.min(0)-0.5
        self.maxXYZ=self.xyz.max(0)+0.5
        self.dy_dr=(self.maxXYZ-self.minXYZ)[1]/self.height
        self.dx_dc=(self.maxXYZ-self.minXYZ)[0]/self.width

        self.window_name=window_name
        self.bg_image=self.draw_all_dots(height_range=[0.1,2.0])
        self.fg_image=self.bg_image

    # Draw the dots with the specified color, returning the image
    #   optionally start with a background image. local_rgb should either be an array of size (N,3)
    #   where N is the same size as local_xyz, or else a tuple of 3 values
    def draw_dots(self, local_xyz:np.array, local_rgb, bg_image=None):
        if bg_image is not None:
            image=copy.copy(bg_image)
        else:
            image=255*np.ones((self.height,self.width,3),dtype=np.uint8)

        rr=np.argsort(local_xyz[:,2])

        if type(local_rgb)==tuple:
            clr=local_rgb
            dynamic_color=False
        elif local_rgb.shape[0]==local_xyz.shape[0]:
            dynamic_color=True
        else:
            raise Exception("Color array is incorrect - not drawing")

        for idx in range(len(rr)):
            try:
                row,col=self.xyz_to_rc(local_xyz[rr[idx]])
                if dynamic_color:
                    tmpC=(local_rgb[rr[idx]]*255).astype(int).tolist()
                    clr=(tmpC[2],tmpC[1],tmpC[0])
                cv2.circle(image, (col, row), radius=2,color=clr,thickness=-1)
            except Exception as e:
                print("Exception: " + str(e))
                print("skipping pixel during dot printing")
        return image

    def draw_all_dots(self, height_range=[-1.0, 100], bg_image=None):
        whichP=np.where((self.xyz[:,2]<height_range[1])*(self.xyz[:,2]>height_range[0]))
        return self.draw_dots(self.xyz[whichP], self.rgb[whichP], bg_image=bg_image)

    def xyz_to_rc(self, xyz_pt):
        row=self.height-int((xyz_pt[1]-self.minXYZ[1])/self.dy_dr)
        col=int((xyz_pt[0]-self.minXYZ[0])/self.dx_dc)
        return row,col

    def rc_to_xy(self, row, col):
        y=(self.height-row)*self.dy_dr+self.minXYZ[1]
        # row=self.height-int((xyz_pt[1]-self.minXYZ[1])/self.dy_dr)
        x=col*self.dx_dc+self.minXYZ[0]
        # col=int((xyz_pt[0]-self.minXYZ[0])/self.dx_dc)
        return x,y
        
    def overlay_dots(self, xyz:np.array, fixed_color=(0,0,255)):
        return self.draw_dots(xyz,local_rgb=fixed_color, bg_image=self.bg_image)
    
    # def add_box_to_fg(self, xyz, fixed_color=(0,0,255)):
    #     minP=xyz.min(0)
    #     maxP=xyz.max(0)
    #     minR,minC=self.xyz_to_rc(minP)
    #     maxR,maxC=self.xyz_to_rc(maxP)
    #     return cv2.rectangle(image,(minC,minR),(maxC,maxR),color=fixed_color,thickness=2)

    def set_fg_image(self, cv_image:np.array):
        self.fg_image=copy.copy(cv_image)

def draw_target_circle(image, target_xyz, fList, key, cam_info:camera_params):
    M=np.matmul(cam_info.rot_matrix,fList.get_pose(key))
    row,col=cam_info.globalXYZ_to_imageRC(target_xyz[0],target_xyz[1],target_xyz[2],M)
    if row<0 or row>=image.shape[0] or col<0 or col>=image.shape[1]:
        print("Target not in image - skipping")
        return image
    radius=int(image.shape[0]/100)
    return cv2.circle(image, (int(col),int(row)), radius=radius, color=(0,0,255), thickness=-1)

def get_closest_point(pts, x, y):
    global image_interface
    closestP=np.argmin(((pts[:,:2]-[x,y])**2).sum(1))
    return pts[closestP]

def visualize_box(fList, pts, row, col, cam_info:camera_params, threshold=0.6):
    global image_interface
    try:
        x,y=image_interface.rc_to_xy(int(row), int(col))
        closestP=get_closest_point(pts,x,y)
        z=closestP[2]
        stats=[]

        for key in fList.keys():
            M=np.matmul(cam_info.rot_matrix,fList.get_pose(key))
            V1=np.matmul(M[:3,:3],[0,0,1])
            V2=[x,y,z]-M[:3,3]
            V2_dist=np.sqrt((V2**2).sum())
            if V2_dist<1.0:
                continue
            angle=np.arccos((V1*V2/V2_dist).sum())
            stats.append([np.abs(angle),V2_dist, key])
        stats=np.array(stats)
        valid_views=np.where(stats<threshold)[0]

        if len(valid_views)>0:        
            rr=valid_views[np.argsort(stats[valid_views,0])]
            fName=fList.get_color_fileName(stats[rr[0],2])
            image=cv2.imread(fName,-1)            
            image=draw_target_circle(image, [x,y,z], fList, stats[rr[0],2], cam_info)
            # fName=fList.get_color_fileName(1137)
            # image=cv2.imread(fName,-1)            
            # image=draw_target_circle(image, [x,y,z], fList, 1137, cam_info)
            if len(valid_views)>1:
                selectedV=np.random.choice(stats[rr[1:],2],2)
                fName2=fList.get_color_fileName(selectedV[0])
                fName3=fList.get_color_fileName(selectedV[1])
                image2=cv2.imread(fName2,-1)
                image2=draw_target_circle(image2, [x,y,z], fList, selectedV[0], cam_info)
                image3=cv2.imread(fName3,-1)
                image3=draw_target_circle(image3, [x,y,z], fList, selectedV[1], cam_info)
                image=np.vstack((image,image2,image3))
                tgt_size=(int(image2.shape[1]/4.0),int(3*image2.shape[0]/4.0))
            else:
                tgt_size=(int(image.shape[1]/4.0),int(image.shape[0]/4.0))
            image=cv2.resize(image,tgt_size)    
            cv2.imshow("views",image)
            cv2.waitKey(1)
    except Exception as e:
        print(e)
        pdb.set_trace()
    

def add_overlay(ply_fileName:str,color=(0,0,255)):
    global image_interface, mouse_clicks
    try:
        pcl_target=o3d.io.read_point_cloud(ply_fileName)
        if pcl_target is None:
            return None
    except Exception as e:
        return None

    tgt_pts=np.array(pcl_target.points)

    # need to filter out NaN's ... not sure why they
    #   are being added to the pcloud, but it causes 
    #   problems downstream
    validP=np.where(np.isnan(tgt_pts).sum(1)==0)
    tgt_pts=tgt_pts[validP]
    image=image_interface.overlay_dots(tgt_pts,color)
    image_interface.set_fg_image(image)
    return tgt_pts

def label_scannet_pcloud(root_dir, raw_dir, save_dir, targets):
    global image_interface, mouse_clicks

    fList=build_file_structure(root_dir+"/"+raw_dir, root_dir+"/"+save_dir)

    # Load a previously created annotation file
    if os.path.exists(fList.get_annotation_file()):
        with open(fList.get_annotation_file(),'r') as handle:
            annotations=json.load(handle)
    else:
        annotations=dict()

    # Load the camera info file
    s_root=root_dir.split('/')
    if s_root[-1]=='':
        par_file=root_dir+"%s.txt"%(s_root[-2])
    else:
        par_file=root_dir+"/%s.txt"%(s_root[-1])
    params=load_camera_info(par_file)

    # Pull the point cloud for drawing a background
    s2=root_dir
    if s2[-1]=="/":
        s2=s2[:-1]

    # Check the existence of the labeled pointclouds
    #   do this here before creating the named window
    #   so that we can exit faster if none of the 
    #   targets are found
    is_valid_target=False
    for tgt in targets:
        ply_label_fileName=fList.get_labeled_pcloud_fileName(tgt)        
        is_valid_target=is_valid_target or os.path.exists(ply_label_fileName)
    if not is_valid_target:
        print("No labeled target files found - saving annotation file and exiting")
        with open(fList.get_annotation_file(),'w') as handle:
            json.dump(annotations, handle)
        return

    # Create the named window        
    window_name=root_dir
    cv2.namedWindow(window_name)
    pcl_raw=o3d.io.read_point_cloud(fList.get_combined_pcloud_fileName())
    image_interface=drawn_image(pcl_raw,window_name=window_name)
    cv2.setMouseCallback(window_name, mouse_cb)

    #Load target pcloud
    for tgt in targets:
        print(tgt)
        # Load both the labeled annotation + the clip pointclouds
        # useful to have both as we can switch between to identify
        # object boundaries
        ply_combo_fileName=fList.get_combined_pcloud_fileName(tgt)
        ply_label_fileName=fList.get_labeled_pcloud_fileName(tgt)        
        tgt_pts_clip=add_overlay(ply_combo_fileName,color=(0,0,255))
        tgt_pts_label=add_overlay(ply_label_fileName,color=(128,0,128))
        # pdb.set_trace()
        if tgt_pts_label is not None and tgt_pts_label.shape[0]>0:
            tgt_pts=tgt_pts_label
            activeP='label'
        # elif tgt_pts_clip is not None:
        #     tgt_pts=tgt_pts_clip
        #     activeP='clip'
        else:
            # print(f"Missing ply files: {ply_combo_fileName} and {ply_label_fileName}")
            print(f"Missing ply files: {ply_label_fileName}")
            continue

        if tgt not in annotations:
            annotations[tgt]=[]
        else:
            # Need to draw the existing annotations
            for annot_ in annotations[tgt]:
                row1,col1=image_interface.xyz_to_rc(annot_[:3])
                row2,col2=image_interface.xyz_to_rc(annot_[3:])
                image=cv2.rectangle(image_interface.fg_image, (col1,row1), (col2,row2), color=(0,255,0))
                image_interface.set_fg_image(image)

        while(1):
            cv2.imshow(window_name, image_interface.fg_image)
            # cv2.imshow(window_name, image_interface.bg_image)
            input_key=cv2.waitKey(0)
            if input_key==ord('n'):
                print("moving to next object")
                with open(fList.get_annotation_file(),'w') as handle:
                    json.dump(annotations, handle)
                break
            elif input_key==ord('a'):   # switch the set of active points between clip + label
                if activeP=='clip':
                    if tgt_pts_label is not None:
                        tgt_pts=tgt_pts_label
                        activeP='label'
                        image=image_interface.overlay_dots(tgt_pts,(128,0,128))
                        image_interface.set_fg_image(image)
                    else:
                        print("Cannot switch set of active points")
                else:
                    if tgt_pts_clip is not None:
                        tgt_pts=tgt_pts_clip
                        activeP='clip'
                        image=image_interface.overlay_dots(tgt_pts,(0,0,255))
                        image_interface.set_fg_image(image)
                    else:
                        print("Cannot switch set of active points")                        
            elif input_key==ord('q'):
                print("exiting labeling task")
                import sys
                sys.exit(-1)
            elif input_key==ord('c'):
                print("Clearing any existing mouse clicks")
                mouse_clicks=[]
            elif input_key==ord('s'):
                if len(mouse_clicks)==2:
                    print("Saving box")
                    image=cv2.rectangle(image_interface.fg_image, mouse_clicks[0], mouse_clicks[1], color=(0,255,0))
                    image_interface.set_fg_image(image)
                    # Now convert the mouse clicks into real world coordinates
                    x1,y1=image_interface.rc_to_xy(mouse_clicks[0][1], mouse_clicks[0][0])
                    x2,y2=image_interface.rc_to_xy(mouse_clicks[1][1], mouse_clicks[1][0])
                    xMin=min(x1,x2)
                    xMax=max(x1,x2)
                    yMin=min(y1,y2)
                    yMax=max(y1,y2)
                    containsP=np.where((tgt_pts[:,0]>=xMin)*(tgt_pts[:,0]<=xMax)*(tgt_pts[:,1]>=yMin)*(tgt_pts[:,1]<=yMax))
                    if len(containsP[0])==0:
                        closestP=get_closest_point(tgt_pts,x1,y1)
                        annotations[tgt].append([xMin,yMin,closestP[2],xMax,yMax,closestP[2]])
                    else:
                        annotations[tgt].append([xMin,yMin,tgt_pts[containsP][:,2].min(),xMax,yMax,tgt_pts[containsP][:,2].max()])
                    mouse_clicks=[]
                else:
                    print("Not enough points to save an annotation")
            elif input_key==ord('v'):
                if len(mouse_clicks)==2:
                    col=(mouse_clicks[0][0]+mouse_clicks[1][0])/2.0
                    row=(mouse_clicks[0][1]+mouse_clicks[1][1])/2.0
                elif len(mouse_clicks)==1:
                    col=mouse_clicks[0][0]
                    row=mouse_clicks[0][1]
                else:
                    print("Need to select at least one point to visualize")
                    continue
                print("visualizing selected box")
                visualize_box(fList, tgt_pts, row, col, params)
            elif input_key==ord('h'):
                print("(c)lear box - deletes the existing mouse clicks")
                print("(s)ave box - saves an existing pair of points as an annotation for the current target object")
                print("(v)isualize box - bring up images related to the selected area")
                print("(n)ext object type - close the current set of annotations and move onto the next target type")
                print("(q)uit - exit the program without saving annotations")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir',type=str,help='location of scannet directory to process')
    parser.add_argument('--raw_dir',type=str,default='raw_output', help='subdirectory containing the color images')
    parser.add_argument('--save_dir',type=str,default='raw_output/save_results', help='subdirectory in which to store the intermediate files')
    parser.add_argument('--targets', type=str, nargs='*', default=None,
                    help='Set of target classes to build point clouds for')
    parser.set_defaults(yolo=True)
    # parser.add_argument('--targets',type=list, nargs='+', default=None, help='Set of target classes to build point clouds for')
    args = parser.parse_args()

    label_scannet_pcloud(args.root_dir, args.raw_dir, args.save_dir, args.targets)
