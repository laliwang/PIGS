import torch
import open3d as o3d
import numpy as np
from cuml.cluster import DBSCAN as cuDBSCAN

class Node:
    
    def __init__(self, mask_list, visible_frame, contained_mask, point_ids, node_info, son_node_info, normal):
        '''
            mask_list: list of masks that is within this cluster
            visible_frame: one-hot vector, 1 if the node appears in the frame
            contained_mask: one-hot vector, 1 if the node is contained by the mask
            point_ids: the corresponding 3D point ids
            node_info: for debugging. The iteration and the index of the node in this iteration
            son_node_info: for debugging. Node infos from the last iteration that are merged into this node
        
        '''
        self.mask_list = mask_list
        self.visible_frame = visible_frame
        self.contained_mask = contained_mask
        self.point_ids = point_ids
        self.node_info = node_info
        self.son_node_info = son_node_info
        self.node_normal = normal


    @ staticmethod
    # 这个应该是每次maskclustering时从旧的节点得到新节点的关键函数 如果要加入法向量 那么每个节点的法向量平均化就要在这里做（直接平均或者聚类平均）
    def create_node_from_list(node_list, node_info):
        mask_list = []
        normal_list = []
        visible_frame = torch.zeros(len(node_list[0].visible_frame), dtype=bool).cuda()
        contained_mask = torch.zeros(len(node_list[0].contained_mask), dtype=bool).cuda()
        point_ids = set()
        son_node_info = set()
        for node in node_list:
            mask_list += node.mask_list
            normal_list += [node.node_normal]
            visible_frame = visible_frame | (node.visible_frame).bool()
            contained_mask = contained_mask | (node.contained_mask).bool()
            point_ids = point_ids.union(node.point_ids)
            son_node_info.add(node.node_info)
        
        # # normal mean by naive mean
        # normal_mean = torch.stack(normal_list).mean(dim=0)
        # dominant_normal = normal_mean / torch.norm(normal_mean) # normalize for normal vector

        # normal mean by max clustering
        normal_mean = torch.stack(normal_list)
        normal_mean_np = normal_mean.cpu().numpy()
        clustering = cuDBSCAN(eps=0.02, min_samples=10)
        labels = clustering.fit(normal_mean_np*10).labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        dominant_label = unique_labels[np.argmax(counts)]
        dominant_cluster = normal_mean_np[labels == dominant_label]
        dominant_normal = np.mean(dominant_cluster, axis=0)
        dominant_normal = dominant_normal / np.linalg.norm(dominant_normal)
        dominant_normal = torch.tensor(dominant_normal).float().cuda()

        return Node(mask_list, visible_frame.float(), contained_mask.float(), point_ids, node_info, son_node_info, dominant_normal)
    
    def get_point_cloud(self, scene_points):
        '''
            return:
                pcld: open3d.geometry.PointCloud object, the point cloud of the node
                point_ids: list of int, the corresponding 3D point ids of the node
        '''
        point_ids = list(self.point_ids)
        points = scene_points[point_ids]
        pcld = o3d.geometry.PointCloud()
        pcld.points = o3d.utility.Vector3dVector(points)
        return pcld, point_ids
