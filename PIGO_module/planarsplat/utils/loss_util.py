import torch
import torch.nn.functional as F

def metric_depth_loss(depth_pred, depth_gt, mask, max_depth=4.0, weight=None):
    depth_mask = torch.logical_and(depth_gt<=max_depth, depth_gt>0)
    depth_mask = torch.logical_and(depth_mask, mask)
    if depth_mask.sum() == 0:
        depth_loss = torch.tensor([0.]).mean().cuda()
    else:
        if weight is None:
            depth_loss = torch.mean(torch.abs((depth_pred - depth_gt)[depth_mask]))
        else:
            depth_loss = torch.mean((weight * torch.abs(depth_pred - depth_gt))[depth_mask])
    return depth_loss

def normal_loss(normal_pred, normal_gt, mask):
    normal_pred = F.normalize(normal_pred, dim=-1)
    normal_gt = F.normalize(normal_gt, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1)[mask].mean()
    cos = (1. - torch.sum(normal_pred * normal_gt, dim=-1))[mask].mean()
    return l1, cos

def mask_rgb_loss(rgb_pred, rgb_gt, mask):
    rgb_pred = rgb_pred.permute(1, 2, 0).reshape(-1, 3)
    if mask.sum() == 0:
        rgb_loss = torch.tensor([0.]).mean().cuda()
    else:
        rgb_loss = torch.mean(torch.abs((rgb_pred - rgb_gt)[mask]))
    return rgb_loss

def distance_link_update(distance_link, render_distance, view_info):
    # distance_link_i 6 dim 分别为
    # [tran_x, tran_y, tran_z, distance_i, w_i, N_i]
    # 前三维构成上一帧相机位置，distance_i为该平面实例增量更新距离，w_i为加权权重，N_i为该平面实例距离更新次数
    view_trans = view_info.pose[:3, 3]
    view_rot = view_info.pose[:3, :3]
    distance_mean = render_distance.squeeze().clone().detach() # (480, 640)
    view_h, view_w = distance_mean.shape
    mask_plane = view_info.mask_mvsa.reshape(view_h, view_w) # (480, 640)
    mask_normal = view_info.mono_normal_local.reshape(view_h, view_w, -1) # (480, 640, 3)
    unique_mask = torch.unique(mask_plane).tolist()

    # return distance link list and distance_mean supervision
    distance_return = torch.zeros_like(distance_mean).detach()

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
        distance_link[int(i-1)] = distance_link_i # 将更新过的distance_link返回到distance_link中

        if distance_link_i[3] != 0:
            distance_return[mask_i] = distance_link_i[3]
        else:
            assert distance_link_i[3] == 0, "Updated distance_link[3] should not be 0!"

    # return updated distance_link and current distance_mean as supervision
    return distance_link, distance_return


def mask_distance_loss(render_distance, distance_mean, mask):
    render_distance = render_distance.squeeze().view(-1)
    distance_mean = distance_mean.detach().squeeze().view(-1)
    distance_loss = torch.abs((render_distance - distance_mean)[mask]).mean()
    return distance_loss