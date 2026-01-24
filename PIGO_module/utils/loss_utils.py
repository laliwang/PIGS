#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np

def l1_loss(network_output, gt, mask_plane=None):
    if mask_plane is not None:
        mask_plane = ~(mask_plane==0.0)
        return torch.abs((network_output-gt)[...,mask_plane]).mean()
    else:
        return torch.abs((network_output - gt)).mean()

def l1_depth_loss(network_output, gt, max_depth, mask_plane=None):
    # 可以改成只在mask内进行深度监督 但是这个无疑会破坏我的平面假设啊 我觉得不行
    gt[gt>max_depth] = 0
    network_output = network_output.squeeze()
    if mask_plane is not None:
        mask_plane = ~(mask_plane==0.0)
        gt[~mask_plane] = 0
    gt_mask=(gt>0)
    depth_loss = torch.abs((network_output[gt_mask] - gt[gt_mask]))
    # depth_loss_normalized = depth_loss / depth_loss.max()
    return depth_loss.mean()

def get_mask_mean_normal(normal,mask):
    num_mask=torch.unique(mask).size(0)-1
    normal_temp=normal.clone()
    for i in range(1,num_mask):
        mask_i= mask==float(i)
        selected_normal_i = normal_temp[:, mask_i]
        mean_values = selected_normal_i.mean(dim=1)
        for c in range(normal_temp.shape[0]):
            normal_temp[c, mask_i] = mean_values[c]
    return normal_temp

def get_mask_mean_distance(distane,mask):
    num_mask=torch.unique(mask).size(0)-1
    distance_temp=distane.clone()
    for i in range(1,num_mask):
        mask_i= mask==float(i)
        selected_distance_i = distance_temp[:, mask_i]
        mean_values = selected_distance_i.mean(dim=1)
        for c in range(distance_temp.shape[0]):
            distance_temp[c, mask_i] = mean_values[c]
    return distance_temp

def plane_loss_normal_l1(normal,mask):
    mask_blank= mask==0.0
    normal_mean=get_mask_mean_normal(normal,mask)
    normal_real=normal.clone()
    normal_real[:, mask_blank]=0.0
    loss= (((normal_real - normal_mean)).abs().sum(0)).mean()
    return loss

def plane_loss_distance_l1(distance,mask):
    mask_blank= mask==0.0
    distance_mean=get_mask_mean_distance(distance,mask)
    distance_real=distance.clone()
    distance_real[:, mask_blank]=0.0
    loss= (((distance_real - distance_mean)).abs().sum(0)).mean()
    return loss

def mask_mean_distance(distance, mask):
    distance_mean = distance.clone().detach()
    unique_mask = torch.unique(mask).tolist()
    for i in unique_mask:
        if i == 0.0:
            continue
        mask_i = (mask == i)
        if mask_i.sum() == 0:
            continue
        mean_distance_i = distance_mean[mask_i].mean()
        distance_mean[mask_i] = mean_distance_i
    return distance_mean


def mask_mean_normal(normal, mask):
    normal_mean = normal.clone().detach()
    unique_mask = torch.unique(mask).tolist()
    for i in unique_mask:
        if i == 0.0:
            continue
        mask_i = (mask == i)
        if mask_i.sum() == 0:
            continue
        mean_normal_i = normal_mean[mask_i].mean(dim=0)
        normal_mean[mask_i] = mean_normal_i
    return normal_mean


def mask_normal_loss_l1(render_normal, mask_normal, mask_plane):
    mask_plane = ~(mask_plane==0.0)
    render_normal = render_normal.permute(1,2,0) # render_nomal: [H,W,C]
    render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=-1)
    render_n = render_normal[mask_plane]
    mask_n = mask_normal[mask_plane]
    loss = torch.abs((render_n - mask_n)).mean()
    return loss

def mask_distance_loss_l1(render_distance, mask_plane):
    render_distance = render_distance.squeeze() # render_distance: [H,W] mask_plane: [H,W]
    with torch.no_grad():
        mask_distance = mask_mean_distance(render_distance, mask_plane)
    device = render_distance.device
    mask_distance = mask_distance.to(device)
    mask_binary = ~(mask_plane==0.0)
    loss = torch.abs((render_distance - mask_distance)[mask_binary]).mean()
    return loss

def mask_alpha_loss_l1(render_alpha, mask_plane):
    render_alpha = render_alpha.squeeze() # render_distance: [H,W] mask_plane: [H,W]
    mask=torch.zeros_like(render_alpha).float()
    device = render_alpha.device
    mask = mask.to(device).detach()

    mask_ones = (~(mask_plane==0.0)).float() # 1.0 的位置代表存在监督mask
    mask_zeros = torch.zeros_like(mask_ones).float()
    kernel = torch.ones((1, 1, 15, 15), device=mask_ones.device)
    mask_id = torch.unique(mask_plane)
    for i in mask_id:
        if i == 0.0:
            continue
        mask_id = (mask_plane==i).float()
        mask_id = torch.nn.functional.conv2d(mask_id.unsqueeze(0).unsqueeze(0), kernel, padding=7)
        mask_id = (mask_id > 0.0).squeeze(0).squeeze(0).float()
        mask_zeros += mask_id
    
    mask_binary_1 = (mask_ones.to(torch.int) == mask_zeros.to(torch.int)) & (mask_ones.to(torch.bool))
    mask_binary_0 = (mask_ones.to(torch.int) != mask_zeros.to(torch.int))
    mask[mask_binary_1] = 1.0
    mask_binary = mask_binary_1 | mask_binary_0

    loss = torch.abs((render_alpha - mask)[mask_binary]).mean()
    return loss

# 和distance_link_update结合使用的distance_loss函数
def mask_distance_loss_link(render_distance, distance_mean, mask_plane, alpha=None):
    render_distance = render_distance.squeeze()
    distance_mean = distance_mean.detach()
    mask_binary = ~(mask_plane==0.0)

    if alpha is not None:
        alpha = alpha.squeeze()
        # distance_mean = distance_mean / alpha
        # distance_normalize = render_distance
        distance_normalize = render_distance / alpha
        mask_binary = mask_binary & (alpha!=0.0) & (distance_mean != 0.0)
    else:
        distance_normalize = render_distance
    loss = torch.abs((distance_normalize - distance_mean)[mask_binary]).mean()
    return loss
    
# 需要在 train_hive.py 中的 with torch.no_grad() 中调用
def distance_link_update(distance_link, render_distance, view_trans, view_rot, mask_normal_com, alpha=None):
    mask_plane = mask_normal_com[..., -1]
    mask_normal = mask_normal_com[..., :-1]
    unique_mask = torch.unique(mask_plane).tolist()
    distance_mean = render_distance.squeeze().clone().detach()
    distance_return = torch.zeros_like(distance_mean).detach()
    if alpha is not None:
        alpha = alpha.squeeze().clone().detach()
        # distance_mean[alpha!=0.0] = distance_mean[alpha!=0.0] / alpha[alpha!=0.0]
        distance_mean = distance_mean / alpha
    for i in unique_mask:
        if i == 0.0:
            continue
        distance_link_i = distance_link[int(i-1)]
        mask_i = (mask_plane == i)
        if mask_i.sum() <= 1: # modified in 20250515 for distance warning when mask_i.sum() = 1
            continue
        mean_distance_i = distance_mean[mask_i].mean()
        mean_normal_i = mask_normal[mask_i].mean(dim=0)
        mean_normal_i /= torch.norm(mean_normal_i)
        mean_normal_i = view_rot @ mean_normal_i
        mask_ratio_i = mask_i.sum() / (640*480)
        w_var_i = -torch.log(distance_mean[mask_i].var())
        w_distance_i = (w_var_i * mask_ratio_i) if w_var_i < 10 else (10*mask_ratio_i)
        
        # 如果方差过大 那么根本不纳入考虑
        if w_distance_i < 0:
            continue

        # 如果N=0 直接mean_distance和w赋值即可
        if distance_link_i[5] == 0:
            distance_link_i[3] = mean_distance_i
        else:
            distance_value = distance_link_i[3] - (distance_link_i[0:3] - view_trans) @ mean_normal_i
            distance_link_i[3] = (distance_value*distance_link_i[4] + mean_distance_i*w_distance_i) / (distance_link_i[4] + w_distance_i)
        # 记录上一次更新的相机位置 当前更新权重 迄今为止更新了多少次
        distance_link_i[0:3] = view_trans
        distance_link_i[4] = distance_link_i[4] + w_distance_i
        distance_link_i[5] += 1
        # 将更新过的distance_link返回到distance_link中
        distance_link[int(i-1)] = distance_link_i

        if distance_link_i[3] != 0:
            # distance_mean[mask_i] = distance_link_i[3]
            distance_return[mask_i] = distance_link_i[3]
        else:
            assert distance_link_i[3] == 0, "Updated distance_link[3] should not be 0!"
    return distance_link, distance_return
        
# # 计算每个高斯上各点法相轴距差异的和
# def gaussian_plane_loss(xyz_gs, normal_gs):
#     xyz_diff = xyz_gs.unsqueeze(1) - xyz_gs.unsqueeze(0) # (N,N,3)
#     dot_products = torch.bmm(xyz_diff, normal_gs.unsqueeze(2)).squeeze(2) #(N,N)
#     abs_dot_products = dot_products.abs()
#     # loss = abs_dot_products.sum() / (xyz_gs.shape[0])
#     loss = abs_dot_products.sum() / (xyz_gs.shape[0] * (xyz_gs.shape[0] - 1))
#     return loss
# 换一种方法最后得到的是世界坐标系下的轴距差异 轴距差异由标准差表征？
def gaussian_plane_loss(xyz_gs, normal_gs):
    dot_products = (xyz_gs * normal_gs).sum(dim=1)
    dot_std = torch.std(dot_products)
    return dot_std

# def mask_normal_loss_l1_mean(render_normal, mask_normal, mask_plane):
#     mask_plane = ~(mask_plane==0.0)
#     render_normal = render_normal.permute(1,2,0) # render_nomal: [H,W,C]
#     render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=-1)
#     with torch.no_grad():
#         mean_normal = mask_mean_normal(render_normal, mask_plane)
#     device = render_normal.device
#     mean_normal = mean_normal.to(device)
#     mean_normal = torch.nn.functional.normalize(mean_normal, p=2, dim=-1)
#     render_n = render_normal[mask_plane]
#     mask_n = mean_normal[mask_plane]
#     loss = torch.abs((render_n - mask_n)).mean()
#     return loss

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask_plane=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    if mask_plane is not None:
        mask_plane = ~(mask_plane==0.0)
        img1 = img1 * mask_plane
        img2 = img2 * mask_plane

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim2(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(0)

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
    return grad_img

def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask