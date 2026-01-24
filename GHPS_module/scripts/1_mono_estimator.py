dependencies = ['torch', 'torchvision']

import os
import torch
import time
import cv2
import sys
import numpy as np
import pandas as pd
import open3d as o3d
from natsort import natsorted
from tqdm import trange, tqdm
import copy
import shutil
import argparse
import trimesh

try:
  from mmcv.utils import Config, DictAction
except:
  from mmengine import Config, DictAction

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Metric3D")
    )
)

from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.unproj_pcd import reconstruct_pcd, save_point_cloud
from mono.utils.tsdf_fusion import OurFuser
metric3d_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Metric3D"))

MODEL_TYPE = {
  'ConvNeXt-Tiny': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convtiny.0.3_150.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth',
  },
  'ConvNeXt-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
  },
  'ViT-Small': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth',
  },
  'ViT-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
  },
  'ViT-giant2': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
    'ckpt_file': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth',
  },
}



def metric3d_convnext_tiny(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ConvNeXt-Tiny']['cfg_file']
  ckpt_file = MODEL_TYPE['ConvNeXt-Tiny']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_convnext_large(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ConvNeXt-Large backbone and Hourglass-Decoder head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ConvNeXt-Large']['cfg_file']
  ckpt_file = MODEL_TYPE['ConvNeXt-Large']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_vit_small(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ViT-Small backbone and RAFT-4iter head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ViT-Small']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-Small']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_vit_large(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ViT-Large backbone and RAFT-8iter head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ViT-Large']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-Large']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def metric3d_vit_giant2(pretrain=False, **kwargs):
  '''
  Return a Metric3D model with ViT-Giant2 backbone and RAFT-8iter head.
  For usage examples, refer to: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
  Args:
    pretrain (bool): whether to load pretrained weights.
  Returns:
    model (nn.Module): a Metric3D model.
  '''
  cfg_file = MODEL_TYPE['ViT-giant2']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-giant2']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    model.load_state_dict(
      torch.hub.load_state_dict_from_url(ckpt_file)['model_state_dict'], 
      strict=False,
    )
  return model

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    grad_img = grad_img*255.0
    return grad_img

def compute_normals(pcd, tran, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)):
    pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_towards_camera_location(camera_location=[tran[0], tran[1], tran[2]])
    return pcd

def project_mask_pc(rgb_input, depth_input, depth_color, pose_input, K_input, max_depth=10, filter_outlier=False, v_size=0.025, depth_scale=1000.0, crop=6):
    frame_id = rgb_input.split('/')[-1].split('.')[0]
    rgb_array = cv2.imread(rgb_input)
    rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
    depth_array = depth_input if isinstance(depth_input, np.ndarray) else cv2.imread(depth_input, -1)
    if max_depth > 0:
        depth_array[depth_array > max_depth] = 0

    # depth need to add black board
    depth_array = cv2.copyMakeBorder(depth_array, crop, crop, crop, crop, cv2.BORDER_CONSTANT, value=0) if crop>0 else depth_array

    if depth_color is not None:
      depth_color_tensor = (torch.from_numpy(depth_color).to(torch.float32)).permute(2, 0, 1)
      grad_depth = get_img_grad_weight(depth_color_tensor).cpu().numpy()
      grad_depth_binary = (grad_depth > 50)

      # a little dilation for binary
      # kernel = np.ones((3,3), np.uint8)
      kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
      grad_depth_binary = (cv2.dilate(grad_depth_binary.astype(np.uint8), kernel, iterations=1))
      grad_depth_binary = cv2.copyMakeBorder(grad_depth_binary, crop, crop, crop, crop, cv2.BORDER_CONSTANT, value=0).astype(np.bool_) if crop>0 else grad_depth_binary.astype(np.bool_)

    #   cv2.imwrite('grad_depth.png', grad_depth)
    #   cv2.imwrite('grad_depth_binary.png', (grad_depth_binary*255.0))
    else:
      grad_depth_binary = np.zeros_like(depth_array).astype(np.bool_)

    fx, fy, cx, cy = K_input[0,0],K_input[1,1],K_input[0,2],K_input[1,2]
    # Convert to 3D coordinates
    mask_valid = (depth_array > 0) & (~grad_depth_binary)
    if depth_color is not None:
      depth_16 = copy.deepcopy(depth_array)
      depth_16[~mask_valid] = 0
      depth_16 = (depth_16*depth_scale).astype(np.uint16)
      cv2.imwrite(rgb_input.replace(f'color/{frame_id}.jpg', f'depth_m3d/{frame_id}.png'), depth_16)

    v, u = np.where(mask_valid)
    depth_mask = depth_array[mask_valid]
    rgb_mask = (rgb_array[mask_valid]).reshape(-1,3)
    x = (u - cx) * depth_mask / fx
    y = (v - cy) * depth_mask / fy
    z = depth_mask
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_mask / 255.0)
    pose = np.loadtxt(pose_input)
    pcd = pcd.transform(pose)
    pcd = pcd.voxel_down_sample(voxel_size=v_size)
    pcd = compute_normals(pcd, pose[:3,3])
    if depth_color is not None:
      depth_color = cv2.copyMakeBorder(depth_color, crop, crop, crop, crop, cv2.BORDER_CONSTANT, value=(0,0,0)) if crop>0 else depth_color
      depth_color[~mask_valid] = 0.0
      return pcd, depth_color
    else:
      return pcd, None

def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def tsdf_fusion(rgb_list, depth_list, pose_list, K_input, depth_factor=1000.0, max_depth=10, voxel_size=0.01, sdf_trunc=0.04):
  # tsdf-volume generation
  volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size, # 体素尺寸
    sdf_trunc=sdf_trunc,  # TSDF 截断距离
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)  # 使用 RGB 彩色数据
  for i in trange(len(depth_list)):
    rgb = (cv2.imread(rgb_list[i])[:, :, ::-1]).astype(np.uint8)
    depth = cv2.imread(depth_list[i], -1)
    pose = np.loadtxt(pose_list[i])
    h, w = depth.shape[:2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(w, h, K_input[0,0], K_input[1,1], K_input[0,2], K_input[1,2])

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb), 
                o3d.geometry.Image(depth),
                depth_scale=depth_factor, 
                depth_trunc=max_depth, 
                convert_rgb_to_intensity=False)
    volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
  # extract mesh from tsdf-volume
  mesh = volume.extract_triangle_mesh()
  mesh.compute_vertex_normals()
  mesh_post = clean_mesh(mesh, min_len=1000)
  mesh_post.remove_unreferenced_vertices()
  mesh_post.remove_degenerate_triangles()
  return mesh, mesh_post

# svo 的 tsdf-fusion 其实未必会比ScalableTSDFVolume更好吧
def tsdf_fusion_svo(rgb_images, depth_images, poses, intrinsic, voxel_size=0.025, sdf_trunc=0.5):
    o3d_device = o3d.core.Device("CUDA:0")
    voxel_block_grid = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=30000,
        device=o3d_device
    )
    for i in trange(len(rgb_images)):
      rgb = (cv2.imread(rgb_images[i])[:, :, ::-1]).astype(np.uint8)
      depth = cv2.imread(depth_images[i], -1)

      rgb_o3d = o3d.t.geometry.Image(rgb).to(o3d_device)
      depth_o3d = o3d.t.geometry.Image(depth).to(o3d_device)

      pose = np.loadtxt(poses[i])
      extrinsic_o3d = o3d.core.Tensor(np.linalg.inv(pose))
      intrinsic_o3d = o3d.core.Tensor(intrinsic[:3,:3])
      block_coords = voxel_block_grid.compute_unique_block_coordinates(depth_o3d, intrinsic_o3d, extrinsic_o3d, 1000.0, 10.0)
      voxel_block_grid.integrate(
          block_coords=block_coords,
          depth=depth_o3d,
          color=rgb_o3d,
          intrinsic=intrinsic_o3d,
          extrinsic=extrinsic_o3d,
          depth_scale=1000.0,  # 深度图像的尺度（单位转换为米）
          depth_max=10.0,  # 最大深度范围
          trunc_voxel_multiplier=sdf_trunc / voxel_size
      )

    mesh = voxel_block_grid.extract_triangle_mesh()
    mesh = mesh.to_legacy()  # 转换为 Open3D 的传统网格格式
    mesh.compute_vertex_normals()
    mesh_post = clean_mesh(mesh, min_len=1000)
    mesh_post.remove_unreferenced_vertices()
    mesh_post.remove_degenerate_triangles()

    return mesh_post

def tsdf_fusion_SR(rgb_list, depth_list, pose_list, K_input, depth_factor=1000.0, max_depth=10, voxel_size=0.01, sdf_trunc=0.04, bounds=None):
  fuser = OurFuser(
            fusion_resolution=voxel_size,
            max_fusion_depth=max_depth,
            num_features=None,
            sdf_trunc=sdf_trunc,
            bounds_gt=bounds,)
  K_input = torch.from_numpy(K_input).unsqueeze(0).cuda()
  
  for i in trange(len(depth_list)):
    depth_path = depth_list[i]
    pose_path = pose_list[i]

    pose = np.loadtxt(pose_path)
    depth_array = cv2.imread(depth_path, -1) / depth_factor # test for metric3d
    # depth_array = np.load(depth_path) # test for depth_pro

    pose = torch.from_numpy(np.linalg.inv(pose)).unsqueeze(0).cuda()
    depth_array = torch.from_numpy(depth_array).unsqueeze(0).unsqueeze(0).cuda()

    fuser.fuse_frames(
              depths_b1hw=depth_array,
              K_b44=K_input,
              cam_T_world_b44=pose,)
  return fuser

def sample_pc_random(mesh_folder, num_samples=200000):
  mesh = trimesh.load(mesh_folder)
  points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
  normals = mesh.face_normals[face_indices]
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  pcd.normals = o3d.utility.Vector3dVector(normals)
  pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points) * 0.5)
  return pcd

def sample_pc_evenly(mesh_folder, num_samples=200000):
  mesh = trimesh.load(mesh_folder)
  areas = mesh.area_faces
  probabilities = areas / areas.sum()
  face_indices = np.random.choice(
      np.arange(len(mesh.faces)), size=num_samples, p=probabilities
  )
  triangles = mesh.triangles[face_indices]

  u = np.sqrt(np.random.rand(num_samples, 1))
  v = np.random.rand(num_samples, 1)
  sampled_points = (
      (1 - u) * triangles[:, 0, :] + 
      (u * (1 - v)) * triangles[:, 1, :] + 
      (u * v) * triangles[:, 2, :]
  )
  normals = mesh.face_normals[face_indices]
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(sampled_points)
  pcd.normals = o3d.utility.Vector3dVector(normals)
  pcd.colors = o3d.utility.Vector3dVector(np.ones_like(sampled_points) * 0.5)
  return pcd

def read_files(directory, endtxt):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(endtxt)]
    file_list = natsorted(file_paths)
    return file_list

def rebuild_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
  #### prepare data
    parser = argparse.ArgumentParser(description='Process scene name.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the input folder.')
    parser.add_argument('--seg_folder', type=str, required=True, help='Path to the output folder.')
    # parser.add_argument('--scannetpp', action="store_true", default=False)
    parser.add_argument("--depth", action="store_true", default=False)
    parser.add_argument("--voxel_size", default=0.025, type=float)
    parser.add_argument("--sdf_trunc", default=0.15, type=float)
    parser.add_argument("--crop", type=int, default=0, help='Crop the image for scannet v2 data.')
    parser.add_argument("--model", type=str, default='m3d', help='model of momocular depth estimation')
    parser.add_argument("--n_sample", default=200000, type=int)
    args = parser.parse_args()
    input_folder = args.data_folder
    output_folder = args.seg_folder
    depth_model = args.model
    print(f'predict depth from scratch: {args.depth}')

    rgb_folder = os.path.join(input_folder, 'color')
    pose_folder = os.path.join(input_folder, 'pose')
    depth_folder = os.path.join(input_folder, 'depth')
    intrinsic_file = os.path.join(input_folder, 'intrinsic/intrinsic_depth.txt')

    depth_m_folder = os.path.join(input_folder, f'depth_{depth_model}')
    mesh_folder = os.path.join(input_folder, 'mesh')
    normal_folder = os.path.join(output_folder, 'normal_npy_m')
    normal_vis_folder = os.path.join(output_folder, 'normal_vis_m')
    points3d_path = os.path.join(mesh_folder, f'points3d_{depth_model}.ply')

    if not os.path.exists(mesh_folder):
        os.makedirs(mesh_folder)

    rgb_list = read_files(rgb_folder, '.jpg')
    pose_list = read_files(pose_folder, '.txt')
    depth_list = read_files(depth_folder, '.png') if os.path.exists(depth_folder) else None

    K_input = np.loadtxt(intrinsic_file)
    gt_depth_scale = 1000.0
    crop = args.crop

    if args.depth:
        # rebuild Metric3D output folder
        rebuild_folder(depth_m_folder)
        rebuild_folder(normal_folder)
        rebuild_folder(normal_vis_folder)
        ### load Metric3d_v2 model before loop
        model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)

        pred_pcd_save = o3d.geometry.PointCloud()
        gt_pcd_save = o3d.geometry.PointCloud()
        with tqdm(total=len(rgb_list)) as pbar:
            for i in range(len(rgb_list)):
                frame_id = rgb_list[i].split('/')[-1].split('.')[0]
                rgb_file = rgb_list[i]
                pose_file = pose_list[i]
                depth_file = depth_list[i] if depth_list is not None else None
                intrinsic = [K_input[0,0], K_input[1,1], K_input[0,2], K_input[1,2]]
                pose_i = np.loadtxt(pose_file)

                abs_rel_err = 0.0

                rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]

                #### ajust input size to fit pretrained model
                input_size = (616, 1064) # for vit model

                # crop for black border
                if crop > 0:
                  intrinsic[2] = intrinsic[2] - crop
                  intrinsic[3] = intrinsic[3] - crop
                  rgb_origin = rgb_origin[crop:-crop, crop:-crop]

                h, w = rgb_origin.shape[:2]
                scale = min(input_size[0] / h, input_size[1] / w)
                rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                # remember to scale intrinsic, hold depth
                intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
                # padding to input_size
                padding = [123.675, 116.28, 103.53]
                h, w = rgb.shape[:2]
                pad_h = input_size[0] - h
                pad_w = input_size[1] - w
                pad_h_half = pad_h // 2
                pad_w_half = pad_w // 2
                rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
                pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

                #### normalize
                mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
                std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
                rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
                rgb = torch.div((rgb - mean), std)
                rgb = rgb[None, :, :, :].cuda()

                ###################### canonical camera space ######################
                # inference
                model.cuda().eval()
                with torch.no_grad():
                    pred_depth, confidence, output_dict = model.inference({'input': rgb})
                
                # un pad
                pred_depth = pred_depth.squeeze().detach()
                pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
                
                # upsample to original size
                pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
                ###################### canonical camera space ######################

                #### de-canonical transform
                canonical_to_real_scale = (intrinsic[0] + intrinsic[1]) / 2000.0 # 1000.0 is the focal length of canonical camera
                pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
                pred_depth = torch.clamp(pred_depth, 0, 300)

                #### you can now do anything with the metric depth 
                # such as evaluate predicted depth
                if depth_file is not None:
                    gt_depth = cv2.imread(depth_file, -1)
                    gt_depth = gt_depth / gt_depth_scale
                    gt_depth = gt_depth[crop:-crop, crop:-crop] if crop > 0 else gt_depth
                    gt_depth = torch.from_numpy(gt_depth).float().cuda()
                    assert gt_depth.shape == pred_depth.shape
                    
                    mask = (gt_depth > 1e-8)
                    abs_rel_err = (torch.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]).mean()
                    abs_rel_err = round(abs_rel_err.item(),4)

                depth_color = pred_depth.cpu().numpy()
                depth_color = (depth_color - depth_color.min()) / (depth_color.max() - depth_color.min() + 1e-20)
                depth_color = (depth_color * 255).clip(0, 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_color, cv2.COLORMAP_JET)

                pred_pcd, depth_color_filter = project_mask_pc(rgb_file, pred_depth.cpu().numpy(), depth_color, pose_file, K_input, crop=crop)
                # gt_pcd, _ = project_mask_pc(rgb_file, gt_depth.cpu().numpy(), None, pose_file, K_input)

                ### normal are also available
                if 'prediction_normal' in output_dict: # only available for Metric3Dv2, i.e. vit model
                    pred_normal = output_dict['prediction_normal'][:, :3, :, :]
                    normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
                    # un pad and resize to some size if needed
                    pred_normal = pred_normal.squeeze()
                    pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]
                    pred_normal = torch.nn.functional.interpolate(pred_normal[None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()

                    pred_conf = normal_confidence.squeeze()
                    pred_conf = pred_conf[pad_info[0] : pred_conf.shape[0] - pad_info[1], pad_info[2] : pred_conf.shape[1] - pad_info[3]]
                    pred_conf = torch.nn.functional.interpolate(pred_conf[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()
                    
                    # you can now do anything with the normal
                    # such as visualize pred_normal
                    pred_normal_np = pred_normal.cpu().numpy().transpose((1, 2, 0))
                    pred_normal_np /= np.linalg.norm(pred_normal_np, axis=-1, keepdims=True)
                    pred_normal_np = cv2.copyMakeBorder(pred_normal_np, crop, crop, crop, crop, cv2.BORDER_CONSTANT, value=(-1,-1,-1)) if crop > 0 else pred_normal_np
                    pred_normal_vis = (pred_normal_np + 1) / 2

                    # pred_conf 的阈值 取 3,4,5 都比较合适
                    pred_conf_np = pred_conf.cpu().numpy()
                    pred_conf_np = cv2.copyMakeBorder(pred_conf_np, crop, crop, crop, crop, cv2.BORDER_CONSTANT, value=0) if crop > 0 else pred_conf_np
                    pred_normal_conf = np.concatenate((pred_normal_np, pred_conf_np[:,:,None]), axis=2).astype(np.float16)
                    np.save(os.path.join(normal_folder, f'{frame_id}.npy'), pred_normal_conf)
                    cv2.imwrite(os.path.join(normal_vis_folder, f'{frame_id}.png'), (pred_normal_vis * 255).astype(np.uint8))

                    pbar.set_postfix(abs_rel_err=f'{abs_rel_err:.4f}')
                    pbar.update(1)

    print("Running tsdf-fusion for mesh extracton......")
    depth_m_list = read_files(depth_m_folder, 'png')

    # get bounds from gt_mesh
    if 'ScanNet++' in input_folder:
        print('For ScanNet++, we use gt_mesh to get bounds')
        gt_mesh_path = os.path.join(input_folder, 'mesh_aligned_0.05.ply')
        gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
        bbox = gt_mesh.get_axis_aligned_bounding_box()
        expansion_factor = 0.1
        bbox_min = bbox.min_bound - expansion_factor
        bbox_max = bbox.max_bound + expansion_factor
        bbox_np = np.concatenate((bbox_min, bbox_max))
    else:
        bbox_np = None

    # tsdf_fusion Simple Recon
    pred_fuser_SR = tsdf_fusion_SR(rgb_list, depth_m_list, pose_list, K_input, voxel_size=args.voxel_size, sdf_trunc=args.sdf_trunc, bounds=bbox_np)
    pred_fuser_SR.export_mesh(os.path.join(mesh_folder, f"tsdf_fusion_SR_{depth_model}.ply"))

    # # randomly sample point from mesh
    # points3d_SR = sample_pc_random(os.path.join(mesh_folder, f"tsdf_fusion_SR_{depth_model}.ply"), num_samples=args.n_sample)
    # o3d.io.write_point_cloud(points3d_path, points3d_SR)

    # # tsdf_fusion svo
    # pred_mesh_svo = tsdf_fusion_svo(rgb_list, depth_m_list, pose_list, K_input, voxel_size=args.voxel_size, sdf_trunc=args.sdf_trunc)
    # o3d.io.write_triangle_mesh(os.path.join(mesh_folder, "tsdf_fusion_svo.ply"), pred_mesh_svo, 
    #             write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
    
    
    