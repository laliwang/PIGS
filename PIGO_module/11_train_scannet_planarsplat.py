import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from random import randint
import numpy as np
from tqdm import tqdm, trange
from natsort import natsorted
from typing import NamedTuple, List, Dict
from loguru import logger
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
import open3d as o3d
import math
import argparse
import cv2
import sys

# from planarsplat
# from planarsplat.run.view_info_scannet import ViewInfoBuilder
from planarsplat.run.view_info_scannet_pigs import ViewInfoBuilder
from planarsplat.run.net_wrapper import PlanarRecWrapper
from planarsplat.utils.misc_util import setup_logging, get_train_param, save_config_files, prepare_folders, get_class
from planarsplat.utils.trainer_util import resume_model, calculate_plane_depth, plot_plane_img, save_checkpoints
from planarsplat.utils.loss_util import normal_loss, metric_depth_loss, mask_rgb_loss, distance_link_update, mask_distance_loss
from planarsplat.utils.mesh_util import get_coarse_mesh
from planarsplat.utils.merge_util import merge_plane

def read_files(directory, endtxt):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(endtxt)]
    file_list = natsorted(file_paths)
    return file_list


# I directly define the PlanarSplatTrainRunner here
class PlanarSplatTrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        self.conf = kwargs['conf']
        self.expname, self.scan_id, self.timestamp, is_continue = get_train_param(kwargs, self.conf)
        self.expdir, self.plane_plots_dir, self.checkpoints_path, self.model_subdir = prepare_folders(kwargs, self.expname, self.timestamp)
        setup_logging(os.path.join(self.expdir, 'train.log'))
        
        logger.info('Shell command : {0}'.format(' '.join(sys.argv)))
        save_config_files(self.expdir, self.conf)

        # =======================================  loading dataset
        logger.info('Loading data...')
        # ViewInfoBuilder 是我们重写的datasetloader 适配我们的数据集格式
        self.dataset = ViewInfoBuilder(kwargs['data_folder'], kwargs['prior_folder'], kwargs['mvsa_folder'])
        self.ds_len = self.dataset.n_images
        self.H = self.dataset.img_res[0]
        self.W = self.dataset.img_res[1]
        logger.info('Data loaded. Frame number = {0}'.format(self.ds_len))

        # =======================================  build plane model
        self.plane_model_conf = self.conf.get_config('plane_model')

        # =======================================  pigs parameters
        self.conf['dataset']['mesh_path'] = self.dataset.mono_mesh_dest
        self.conf['dataset']['mvsa_path'] = self.dataset.mvsa_pts_path
        self.conf['dataset']['color_path'] = self.dataset.color_path
        self.distance_link = torch.zeros(self.dataset.inst_num, 6).float().cuda()

        net = PlanarRecWrapper(self.conf, self.plane_plots_dir)
        self.net = net.cuda()
        self.resumed = False
        self.start_iter = resume_model(self) if is_continue else 0
        self.iter_step = self.start_iter
        self.net.build_optimizer_and_LRscheduler()

        # ======================================= plot settings
        self.do_vis = kwargs['do_vis']
        self.plot_freq = self.conf.get_int('train.plot_freq')        
        
        # ======================================= loss settings
        loss_plane_conf = self.conf.get_config('plane_model.plane_loss')
        self.weight_plane_normal = loss_plane_conf.get_float('weight_mono_normal')
        self.weight_plane_depth = loss_plane_conf.get_float('weight_mono_depth')

        # ======================================= training settings
        self.max_total_iters = self.conf.get_int('train.max_total_iters')
        self.process_plane_freq_ite = self.conf.get_int('train.process_plane_freq_ite')
        self.coarse_stage_ite = self.conf.get_int('train.coarse_stage_ite')
        self.split_start_ite = self.conf.get_int('train.split_start_ite')
        self.check_vis_freq_ite = self.conf.get_int('train.check_plane_vis_freq_ite')
        self.data_order = self.conf.get_string('train.data_order')
    

    def train(self):
        logger.info("Training...")
        if self.start_iter >= self.max_total_iters:
            return
        weight_decay_list = []
        for i in tqdm(range(self.max_total_iters+1), desc="generating sampling idx list..."):
            weight_decay_list.append(max(math.exp(-i / self.max_total_iters), 0.1))
        logger.info('Start training at {:%Y_%m_%d_%H_%M_%S}'.format(datetime.now()))
        self.net.train()

        view_info_list = None
        progress_bar = tqdm(range(self.start_iter, self.max_total_iters+1), desc="Training progress")
        calculate_plane_depth(self)
        for iter in range(self.start_iter, self.max_total_iters + 1):
            self.iter_step = iter
            # ======================================= process planes
            if iter > self.coarse_stage_ite and iter % self.process_plane_freq_ite==0:  
                self.net.regularize_plane_shape()
                self.net.prune_small_plane()
                if iter > self.split_start_ite and iter <= self.max_total_iters - 1000:
                    logger.info('splitting...')
                    ori_num = self.net.planarSplat.get_plane_num()
                    self.net.split_plane()
                    new_num = self.net.planarSplat.get_plane_num()
                    logger.info(f'plane num: {ori_num} ---> {new_num}')
            # ======================================= get view info
            if not view_info_list:
                view_info_list = self.dataset.view_info_list.copy()
            if self.data_order == 'rand':
                view_info = view_info_list.pop(randint(0, len(view_info_list)-1))
            else:
                view_info = view_info_list.pop(0)
            raster_cam_w2c = view_info.raster_cam_w2c
            # ======================================= zero grad
            self.net.optimizer.zero_grad()
            #  ======================================= plane forward
            rendered_rgb, allmap = self.net.planarSplat(view_info,iter, return_rgb=True)
            # ------------ get rendered maps
            depth = allmap[0:1].squeeze().view(-1)
            normal_local_ = allmap[2:5]
            normal_global = (normal_local_.permute(1,2,0) @ (raster_cam_w2c[:3,:3].T)).view(-1, 3)
            # ------------ get aux maps
            vis_weight = allmap[1:2].squeeze().view(-1)
            valid_ray_mask = vis_weight > 0.00001
            valid_normal_mask = view_info.mono_normal_global.abs().sum(dim=-1) > 0
            valid_depth_mask = view_info.mono_depth.abs() > 0
            # ------------ use instance-level splatting or not
            if iter > self.coarse_stage_ite:
                with torch.no_grad():
                    valid_mvsa_mask = (self.net.planarSplat.instance_splat_mask(view_info, iter)) > 0
            else:
                valid_mvsa_mask = view_info.mask_mvsa > 0
            valid_ray_mask = valid_ray_mask & valid_depth_mask & valid_normal_mask & valid_mvsa_mask

            # ======================================= calculate losses
            loss_final = 0.
            decay = weight_decay_list[iter]
            # ------------ calculate plane loss
            loss_plane_normal_l1, loss_plane_normal_cos = normal_loss(normal_global, view_info.mono_normal_global, valid_ray_mask)
            loss_plane_depth = metric_depth_loss(depth, view_info.mono_depth, valid_ray_mask, max_depth=10.0)

            # plane losses adopted from pigs
            loss_mask_rgb = mask_rgb_loss(rendered_rgb, view_info.rgb, valid_ray_mask) # mask_rgb loss
            rendered_distance = self.compute_distance_from_dn(allmap[0:1], allmap[2:5], view_info.intrinsic) # shape=(1, 480, 640)
            if iter > self.coarse_stage_ite:
                with torch.no_grad():
                    # update current distance_link list for each plane instance
                    self.distance_link, distance_mean = distance_link_update(self.distance_link, rendered_distance, view_info)
                loss_mask_distance = mask_distance_loss(rendered_distance, distance_mean, valid_ray_mask)
            else:
                loss_mask_distance = 0.0

            loss_plane = (loss_plane_depth * 1.0) * self.weight_plane_depth \
                        + (loss_plane_normal_l1 + loss_plane_normal_cos) * self.weight_plane_normal \
                        + loss_mask_rgb * 1.0 + loss_mask_distance * 1.0
            loss_final += loss_plane * decay

            # ======================================= backward & update plane denom & update learning rate
            loss_final.backward()
            self.net.optimizer.step()
            self.net.update_grad_stats()
            self.net.regularize_plane_shape(False)
            image_index = view_info.index
            self.dataset.view_info_list[image_index].plane_depth = depth.detach().clone()

            with torch.no_grad():
                # Progress bar
                plane_num = self.net.planarSplat.get_plane_num()
                if iter % 10 == 0:
                    loss_dict = {
                        "Planes": f"{plane_num}",
                    }
                    progress_bar.set_postfix(loss_dict)
                    progress_bar.update(10)
                if iter == self.max_total_iters:
                    progress_bar.close()
            
            # ======================================= plot model outputs
            if self.do_vis and iter % self.plot_freq == 0:
                self.net.regularize_plane_shape()
                self.net.eval()
                self.net.planarSplat.draw_plane(epoch=iter, color_path=self.conf['dataset']['color_path'])
                plot_plane_img(self)
                self.net.train()
            
        save_checkpoints(self, iter=self.iter_step, only_latest=False)
        np.savetxt(os.path.join(self.expdir, 'planeids.txt'), self.net.planarSplat.get_plane_ids.detach().cpu().numpy())

    def render(self):
        logger.info("Rendering...")
        self.renderdir = os.path.join(self.expdir, 'renders')
        render_rgb_dir = os.path.join(self.renderdir, 'rgb')
        render_normal_dir = os.path.join(self.renderdir, 'normal')
        render_depth_dir = os.path.join(self.renderdir, 'depth')
        render_distance_dir = os.path.join(self.renderdir, 'distance')
        render_depth16_dir = os.path.join(self.renderdir, 'depth16')
        os.makedirs(render_rgb_dir, exist_ok=True)
        os.makedirs(render_normal_dir, exist_ok=True)
        os.makedirs(render_depth_dir, exist_ok=True)
        os.makedirs(render_distance_dir, exist_ok=True)
        os.makedirs(render_depth16_dir, exist_ok=True)

        self.net.eval()

        with torch.no_grad():
            for view_info in tqdm(self.dataset.view_info_list, desc="Rendering"):
                rendered_rgb, allmap = self.net.planarSplat(view_info, iter=self.iter_step, return_rgb=True)
                with torch.no_grad():
                    valid_mvsa_mask = (self.net.planarSplat.instance_splat_mask(view_info, iter=self.iter_step)) > 0
                valid_mvsa_mask = valid_mvsa_mask.squeeze().reshape(self.H, self.W).detach().cpu().numpy().astype(np.bool_)
                # ------------ get rendered maps
                rendered_depth = allmap[0:1].squeeze().view(-1) # allmap[0:1].shape=(1, 480, 640)
                normal_local_ = allmap[2:5] # allmap[2:5].shape=(3, 480, 640)
                rendered_distance = self.compute_distance_from_dn(allmap[0:1], allmap[2:5], view_info.intrinsic) # shape=(1, 480, 640)

                rendered_normal_local = (normal_local_.permute(1,2,0)).view(-1, 3)
                rendered_normal_np = (F.normalize(rendered_normal_local, dim=-1).reshape(self.H, self.W, 3)).detach().cpu().numpy().astype(np.float16)
                rendered_normal_np[~valid_mvsa_mask] = 0.0
                rendered_normal_color = ((rendered_normal_np + 1.0) / 2.0 * 255).astype(np.uint8)
                
                rendered_rgb_np = (rendered_rgb.permute(1,2,0) * 255).detach().cpu().numpy().astype(np.uint8)
                
                rendered_depth_np = rendered_depth.reshape(self.H, self.W).detach().cpu().numpy()
                rendered_depth_np[~valid_mvsa_mask] = 0.0
                rendered_depth_16 = (rendered_depth_np * 1000.0).astype(np.uint16) # save for next step
                rendered_depth_color = (rendered_depth_np - rendered_depth_np.min()) / (rendered_depth_np.max() - rendered_depth_np.min() + 1e-20)
                rendered_depth_color = (rendered_depth_color * 255).clip(0, 255).astype(np.uint8)
                rendered_depth_color = cv2.applyColorMap(rendered_depth_color, cv2.COLORMAP_JET)
                
                rendered_distance_np = rendered_distance.squeeze().detach().cpu().numpy()
                rendered_distance_color = (rendered_distance_np - rendered_distance_np.min()) / (rendered_distance_np.max() - rendered_distance_np.min() + 1e-20)
                rendered_distance_color = (rendered_distance_color * 255).clip(0, 255).astype(np.uint8)
                rendered_distance_color = cv2.applyColorMap(rendered_distance_color, cv2.COLORMAP_JET)

                image_name = view_info.image_path.split('/')[-1].split('.')[0]
                cv2.imwrite(os.path.join(render_rgb_dir, image_name + ".png"), rendered_rgb_np)
                cv2.imwrite(os.path.join(render_normal_dir, image_name + ".png"), rendered_normal_color)
                np.save(os.path.join(render_normal_dir, image_name + ".npy"), rendered_normal_np)
                cv2.imwrite(os.path.join(render_depth16_dir, image_name + ".png"), rendered_depth_16)
                cv2.imwrite(os.path.join(render_depth_dir, image_name + ".png"), rendered_depth_color)
                # cv2.imwrite(os.path.join(render_distance_dir, image_name + ".png"), rendered_distance_color)

    def merger(self, save_mesh=True):
        logger.info("Merging 3D planar primitives...")
        output_dir = self.conf.get_string('train.rec_folder_name', default='')
        if len(output_dir) == 0:
            output_dir = self.expdir
        self.net.eval()
        save_root = os.path.join(output_dir, f'{self.scan_id}')
        os.makedirs(save_root, exist_ok=True)

        ## prune planes whose maximum radii lower than the threshold
        self.net.prune_small_plane(min_radii=0.02 * self.net.planarSplat.pose_cfg.scale)
        logger.info("number of 3D planar primitives = %d"%(self.net.planarSplat.get_plane_num()))

        coarse_mesh = get_coarse_mesh(
            self.net, 
            self.dataset.view_info_list.copy(), 
            self.H, 
            self.W, 
            voxel_length=0.02, 
            sdf_trunc=0.08)
        
        merge_config_coarse = self.conf.get_config('merge_coarse', default=None)
        merge_config_fine = self.conf.get_config('merge_fine', default=None)
        if merge_config_coarse is not None:
            logger.info(f'mergeing (coarse)...')
            planarSplat_eval_mesh, plane_ins_id_new = merge_plane(
                self.net, 
                coarse_mesh, 
                plane_ins_id=None,
                **merge_config_coarse)
            print(plane_ins_id_new.shape)
            if merge_config_fine is not None:
                logger.info(f'mergeing (fine)...')
                planarSplat_eval_mesh, plane_ins_id_new = merge_plane(
                    self.net, 
                    coarse_mesh, 
                    plane_ins_id=plane_ins_id_new,
                    **merge_config_fine)
        else:
            raise ValueError("No merge configuration found!")
        
        if save_mesh:
            save_path = os.path.join(save_root, f"{self.scan_id}_planar_mesh.ply")
            logger.info(f'saving final planar mesh to {save_path}')
            o3d.io.write_triangle_mesh(
                        save_path, 
                        planarSplat_eval_mesh)
        return planarSplat_eval_mesh
    

    def compute_distance_from_dn(
        self,
        depth_pred: torch.Tensor,
        normal_pred: torch.Tensor,
        K_input: torch.Tensor,
        max_depth: float = 10.0
    ):
        """
        depth_pred : (1, 480, 640)
        normal_pred: (3, 480, 640)
        K_input    : (4, 4)
        return     : distance_pred (1, 480, 640)
        """
        device = depth_pred.device
        dtype = depth_pred.dtype
        depth = depth_pred.clone()
        if max_depth > 0:
            depth = torch.where(depth > max_depth, torch.zeros_like(depth), depth)

        fx = K_input[0, 0]
        fy = K_input[1, 1]
        cx = K_input[0, 2]
        cy = K_input[1, 2]

        _, H, W = depth.shape
        v, u = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij"
        )

        mask_valid = depth[0] > 0  # (H, W)
        z = depth[0]                                   # (H, W)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = torch.stack([x, y, z], dim=-1)        # (H, W, 3)
        normals = normal_pred.permute(1, 2, 0)         # (H, W, 3)
        # distance = - <p, n>
        distance = -torch.sum(points * normals, dim=-1)  # (H, W)
        distance = torch.where(mask_valid, distance, torch.zeros_like(distance))

        return distance.unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='', help='path to orginal data folder')
    parser.add_argument('--prior_folder', type=str, default='', help='path to structual prior folder')
    parser.add_argument('--mvsa_folder', type=str, default='', help='path to mvsa folder')
    parser.add_argument('--base_conf', type=str, default='', help='path to base conf file')
    parser.add_argument('--scene_conf', type=str, default='', help='path to scene conf file')
    args = parser.parse_args()

    data_folder = args.data_folder
    prior_folder = args.prior_folder
    mvsa_folder = args.mvsa_folder
    base_conf_path = os.path.abspath(args.base_conf)
    scene_conf_path = os.path.abspath(args.scene_conf)
    scan_id = os.path.basename(data_folder)

    # 读取planarsplatting confs
    base_conf = ConfigFactory.parse_file(base_conf_path)
    scene_conf = ConfigFactory.parse_file(scene_conf_path)
    cfg = ConfigTree.merge_configs(base_conf, scene_conf)

    # get name of experiment folder
    exps_folder_name = cfg.get_string('train.exps_folder_name', default='exps_result')
    print('exps_folder_name:', exps_folder_name)
    
    # initialize PlanarSplatTrainRunner
    runner = PlanarSplatTrainRunner(
        conf=cfg,
        batch_size=1,
        exps_folder_name=exps_folder_name,
        is_continue=False,
        timestamp='latest',
        checkpoint='latest',
        do_vis=True,
        scan_id=scan_id,
        data_folder=data_folder,
        prior_folder=prior_folder,
        mvsa_folder=mvsa_folder
    )

    runner.train()
    runner.render()
    runner.merger()
    # view_info_list = ViewInfoBuilder(data_folder, prior_folder)
