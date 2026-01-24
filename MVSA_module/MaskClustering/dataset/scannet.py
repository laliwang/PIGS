import open3d as o3d
import numpy as np
import os
import cv2
from natsort import natsorted
from evaluation.constants import SCANNET_LABELS, SCANNET_IDS

class ScanNetDataset:

    def __init__(self, data_folder, seg_folder, render, model, mask) -> None:
        self.root = data_folder
        self.root_seg = seg_folder
        self.rgb_dir = f'{self.root}/color'

        if render:
            self.depth_dir = f'{self.root}/depth_{model}'
            self.point_cloud_path = f'{self.root}/mesh/points3d_{model}_proj.ply'
        else:
            self.depth_dir = f'{self.root}/depth_{model}'
            self.point_cloud_path = f'{self.root}/mesh/points3d_{model}.ply'

        print(f'Using GHPS point :{self.point_cloud_path}')

        self.segmentation_dir = f'{seg_folder}/ghps_output/mask_normal_com_{mask}'
        
        self.object_dict_dir = f'{self.root_seg}/mvsa_output/object'     
        self.mesh_path = self.point_cloud_path
        self.extrinsics_dir = f'{self.root}/pose'
        self.intrinsic_dir = f'{self.root}/intrinsic'

        self.depth_scale = 1000.0
        self.image_size = (640, 480)
    

    def get_frame_list(self, stride):
        image_list = os.listdir(self.rgb_dir)
        image_list = natsorted(image_list, key=lambda x: int(x.split('.')[0]))
        frame_id_list = [int(image_list[i].split('.')[0]) for i in range(0, len(image_list))]
        return frame_id_list
        # end = int(image_list[-1].split('.')[0]) + 1
        # frame_id_list = np.arange(0, end, stride)
        # return list(frame_id_list)
    

    def get_intrinsics(self, frame_id):
        intrinsic_path = f'{self.intrinsic_dir}/intrinsic_depth.txt'
        intrinsics = np.loadtxt(intrinsic_path)

        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinisc_cam_parameters.set_intrinsics(self.image_size[0], self.image_size[1], intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        return intrinisc_cam_parameters
    

    def get_extrinsic(self, frame_id):
        pose_path = os.path.join(self.extrinsics_dir, str(frame_id) + '.txt')
        if not os.path.exists(pose_path):
            pose_path = os.path.join(self.extrinsics_dir, f'{frame_id:06}' + '.txt')
        pose = np.loadtxt(pose_path)
        return pose
    

    def get_depth(self, frame_id):
        depth_path = os.path.join(self.depth_dir, str(frame_id) + '.png')
        if not os.path.exists(depth_path):
            depth_path = os.path.join(self.depth_dir, f'{frame_id:06}' + '.png')
        depth = cv2.imread(depth_path, -1)
        depth = depth / self.depth_scale
        depth = depth.astype(np.float32)
        return depth


    def get_rgb(self, frame_id, change_color=True):
        rgb_path = os.path.join(self.rgb_dir, str(frame_id) + '.jpg')
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(self.rgb_dir, f'{frame_id:06}' + '.jpg')
        rgb = cv2.imread(rgb_path)

        if change_color:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb    


    def get_segmentation(self, frame_id, align_with_depth=False):
        segmentation_path = os.path.join(self.segmentation_dir, f'{frame_id}.npy')
        if not os.path.exists(segmentation_path):
            segmentation_path = os.path.join(self.segmentation_dir, f'{frame_id:06}.npy')
            if not os.path.exists(segmentation_path):
                assert False, f"Segmentation not found: {segmentation_path}"
        segmentation_normal = np.load(segmentation_path)
        segmentation = segmentation_normal[...,-1].astype(np.int32)
        normal = segmentation_normal[...,:-1].astype(np.float16)
        pose = self.get_extrinsic(frame_id)
        if np.sum(np.isinf(pose)) == 0:
            R = pose[:3, :3]
            normal = (np.dot(normal.reshape(-1,3), R.T)).reshape(segmentation.shape[0], segmentation.shape[1], 3)
        if align_with_depth:
            segmentation = cv2.resize(segmentation, self.image_size, interpolation=cv2.INTER_NEAREST)
        return segmentation, normal


    def get_frame_path(self, frame_id):
        rgb_path = os.path.join(self.rgb_dir, str(frame_id) + '.jpg')
        segmentation_path = os.path.join(self.segmentation_dir, f'{frame_id}.png')
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(self.rgb_dir, f'{frame_id:06}' + '.jpg')
        if not os.path.exists(segmentation_path):
            segmentation_path = os.path.join(self.segmentation_dir, f'{frame_id:06}' + '.png')
        return rgb_path, segmentation_path
    

    def get_label_features(self):
        label_features_dict = np.load(f'data/text_features/scannet.npy', allow_pickle=True).item()
        return label_features_dict


    def get_scene_points(self):
        mesh = o3d.io.read_point_cloud(self.point_cloud_path)
        vertices = np.asarray(mesh.points)
        return vertices
    
    
    def get_label_id(self):
        self.class_id = SCANNET_IDS
        self.class_label = SCANNET_LABELS

        self.label2id = {}
        self.id2label = {}
        for label, id in zip(self.class_label, self.class_id):
            self.label2id[label] = id
            self.id2label[id] = label

        return self.label2id, self.id2label