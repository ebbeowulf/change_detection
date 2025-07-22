from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
import pdb
from typing import Optional
from nerfstudio.utils import colormaps
import mediapy as media
import argparse
from nerfstudio.utils.eval_utils import eval_setup
import cv2

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def _get_fname(filepath: Path, data_dir: Path) -> Path:
    """Get the filename of the image file.
    filepath: the base file name of the transformations.
    data_dir: the directory of the data that contains the transform file
    """
    return data_dir / filepath

def load_from_json(filename):
    with open(filename, "r") as fin:
        return json.load(fin)
    
class build_camera_json():
    def __init__(self, transforms_json:str, dataparser_transforms:str):
        self.load_transforms(transforms_json)
        self.build_transform_matrix(dataparser_transforms, self.meta)
        self.parse_frames()

    def add_mat_zeros(self, mat_out):
        return torch.cat([mat_out,torch.tensor([[0,0,0,1]],dtype=mat_out.dtype)],0)

    def build_transform_matrix(self, dataparser_transforms, meta):
        # Need to load the known transform, and then adjust it for
        #   transforms already taken in the conversion to transforms.json
        with open(dataparser_transforms,'r') as fin:
            tf_json=json.load(fin)

        transform_matrix=self.add_mat_zeros(torch.tensor(tf_json['transform'],dtype=torch.float32))
        if "applied_transform" in meta:
            applied_transform = self.add_mat_zeros(torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype))
            self.transform_matrix = transform_matrix @ torch.linalg.inv(applied_transform)
        else:
            self.transform_matrix = transform_matrix

        self.scale_factor=tf_json['scale']
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            self.scale_factor /= applied_scale

    def load_transforms(self, transforms_json: str):
        p_trans=Path(transforms_json)
        # Loading a transforms.json
        if p_trans.suffix == ".json":
            self.meta = load_from_json(p_trans)
            self.data_dir = p_trans.parent
        else:
            self.meta = load_from_json(p_trans / "transforms.json")
            self.data_dir = p_trans
        
    def parse_frames(self):
        self.image_filenames = []
        self.poses = []

        self.height_fixed = self.meta["h"]
        self.width_fixed = self.meta["w"]
        self.fisheye_crop_radius = self.meta.get("fisheye_crop_radius", None)

        # sort the frames by fname
        fnames = []
        for frame in self.meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = _get_fname(filepath, self.data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [self.meta["frames"][ind] for ind in inds]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = _get_fname(filepath, self.data_dir)

            self.image_filenames.append(fname)
            self.poses.append(np.array(frame["transform_matrix"]))

    def get_world_coordinates(self, name_filter:str=None):
        poses = torch.from_numpy(np.array(self.poses).astype(np.float32))
        poses = self.transform_matrix @ poses

        # Scale poses
        poses[:, :3, 3] *= self.scale_factor

        # Filter names
        files = np.array([ str(file) for file in self.image_filenames ])
        if name_filter is not None:
            i_export=[]
            for idx, file in enumerate(files):
                if name_filter in file:
                    i_export.append(idx)                 
            poses=poses[i_export]
            files=files[i_export]


        return poses, files
    
    def get_distortion_params(self):
        k1=float(self.meta["k1"]) if "k1" in self.meta else 0.0
        k2=float(self.meta["k2"]) if "k2" in self.meta else 0.0
        k3=float(self.meta["k3"]) if "k3" in self.meta else 0.0
        k4=float(self.meta["k4"]) if "k4" in self.meta else 0.0
        p1=float(self.meta["p1"]) if "p1" in self.meta else 0.0
        p2=float(self.meta["p2"]) if "p2" in self.meta else 0.0
        return torch.Tensor([k1, k2, k3, k4, p1, p2])
    
    def get_cameras(self, name_filter:str=None):
        poses, image_filenames=self.get_world_coordinates(name_filter)
        if "camera_model" in self.meta:
            camera_type = CAMERA_MODEL_TO_TYPE[self.meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fisheye_crop_radius = self.meta.get("fisheye_crop_radius", None)
        metadata = {"fisheye_crop_radius": fisheye_crop_radius} if fisheye_crop_radius is not None else None
        cameras = Cameras(
            fx=self.meta['fl_x'],
            fy=self.meta['fl_y'],
            cx=self.meta['cx'],
            cy=self.meta['cy'],
            distortion_params=self.get_distortion_params(),
            height=self.height_fixed,
            width=self.width_fixed,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            metadata=metadata
        )
        return cameras, image_filenames

    # Create a camera path for use by ns-render - ignores camera intrinsics
    def generate_camera_path(self, name_filter:str=None):
        poses, files = self.get_world_coordinates(name_filter)
        poses=poses.numpy()
        out = {
            'camera_type': 'perspective',
            'render_height': self.height_fixed,
            'render_width': self.width_fixed,
            'seconds': poses.shape[0],
            'camera_path': []
            }
        for idx in range(poses.shape[0]):
            out['camera_path'].append({'camera_to_world': poses[idx].tolist(), 'fov': 50, 'aspect': 1, 'file_path': files[idx]})
        return out

def synthesize_images(config_path:str, output_path:str, cam_dataset:build_camera_json, image_type='rgb', name_filter='rgb'):
    config, pipeline, _, _ = eval_setup(
        Path(config_path),
        eval_num_rays_per_chunk=None,
        test_mode="inference",
        update_config_callback=None,
    )
    output_dir=Path(output_path)
    cameras, image_filenames=cam_dataset.get_cameras(name_filter)
    for camera_idx in range(cameras.shape[0]):
        # Looks like there must be a squeeze function in the Cameras definition function
        #   Need to undo
        c2=cameras[camera_idx]
        if len(cameras[camera_idx].camera_to_worlds.shape)<3:
            c2.camera_to_worlds=torch.unsqueeze(cameras[camera_idx].camera_to_worlds,dim=0)

        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(c2)
            # outputs = pipeline.model.get_outputs_for_camera(cameras[camera_idx])

        # Figure out the filename
        file=Path(image_filenames[camera_idx])

        # Generate the image from the outputs and save
        if image_type=='rgb' or image_type=='all':
            fName="rgb_" + file.stem.split('_')[-1] + ".png"
            print(fName)
            outFile=output_dir / fName
            output_image = (
                colormaps.apply_colormap(
                    image=outputs['rgb'],
                    colormap_options=colormaps.ColormapOptions(),
                )
                .cpu()
                .numpy()
            )
            media.write_image(outFile, output_image, fmt="png")
        if image_type=='depth' or image_type=='all':
            if 'depth' in outputs:
                fName="depth_" + file.stem.split('_')[-1] + ".png"
                print(fName)
                outFile=output_dir / fName
                output_image=((outputs['depth'].cpu().numpy()/cam_dataset.scale_factor)*1000.0).astype(np.uint16)
                # have to use opencv to save 16 bit
                cv2.imwrite(str(outFile), output_image)
        if image_type=='expected_depth' or image_type=='all':
            if 'expected_depth' in outputs:
                fName="depthE_" + file.stem.split('_')[-1] + ".png"
                print(fName)
                outFile=output_dir / fName
                output_image=((outputs['expected_depth'].cpu().numpy())*1000.0).astype(np.uint16)
                # have to use opencv to save 16 bit
                cv2.imwrite(str(outFile), output_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('nerfacto_dir',type=str,help='location of nerfactor directory containing config.yml and dapaparser_transforms.json')
    parser.add_argument('transforms_file',type=str,help='location of the transforms file to render')
    parser.add_argument('output_dir',type=str,help='where to save the resulting images')
    parser.add_argument('--image-type', type=str, default='all', help="rgb or depth images (default=all)")
    parser.add_argument('--name-filter', type=str, default='rgb', help="name filter to apply to transforms (default=rgb)")
    args = parser.parse_args()

    tf=build_camera_json(args.transforms_file,
                    args.nerfacto_dir + "/dataparser_transforms.json")
    synthesize_images(args.nerfacto_dir + "/config.yml",
                      args.output_dir,
                      tf,
                      image_type=args.image_type,
                      name_filter=args.name_filter)

