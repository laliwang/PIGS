from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import trimesh
import sys

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from tsdf import TSDF, TSDFFuser

class DepthFuser:
    def __init__(
        self,
        fusion_resolution: float = 0.04,
        max_fusion_depth: float = 3.0,
    ):
        self.fusion_resolution = fusion_resolution
        self.max_fusion_depth = max_fusion_depth

class OurFuser(DepthFuser):
    def __init__(
        self,
        fusion_resolution: float = 0.04,
        max_fusion_depth: float = 3,
        num_features: int = 2,
        sdf_trunc: float = 0.2,
        bounds_gt: np.ndarray = None,
    ):
        super().__init__(
            fusion_resolution=fusion_resolution,
            max_fusion_depth=max_fusion_depth,
        )

        if bounds_gt is None:
            bounds = {}
            bounds["xmin"] = -10.0
            bounds["xmax"] = 10.0
            bounds["ymin"] = -10.0
            bounds["ymax"] = 10.0
            bounds["zmin"] = -10.0
            bounds["zmax"] = 10.0
        else:
            bounds = {}
            bounds["xmin"] = bounds_gt[0]
            bounds["ymin"] = bounds_gt[1]
            bounds["zmin"] = bounds_gt[2]
            bounds["xmax"] = bounds_gt[3]
            bounds["ymax"] = bounds_gt[4]
            bounds["zmax"] = bounds_gt[5]

        tsdf_pred = TSDF.from_bounds(
            bounds, voxel_size=fusion_resolution, num_features=num_features
        )

        self.tsdf_fuser_pred = TSDFFuser(tsdf_pred, max_depth=max_fusion_depth, sdf_trunc=sdf_trunc)

    def fuse_frames(
        self,
        depths_b1hw: torch.Tensor,
        K_b44: torch.Tensor,
        cam_T_world_b44: torch.Tensor,
    ):
        self.tsdf_fuser_pred.integrate_depth(
            depth_b1hw=depths_b1hw.half(),
            cam_T_world_T_b44=cam_T_world_b44.half(),
            K_b44=K_b44.half(),
        )

    def fuse_frames_features(self, depths_b1hw, K_b44, cam_T_world_b44, features_bchw):
        self.tsdf_fuser_pred.integrate_depth(
            depth_b1hw=depths_b1hw.half(),
            cam_T_world_T_b44=cam_T_world_b44.half(),
            K_b44=K_b44.half(),
            features_bchw=features_bchw.half(),
        )

    def export_mesh(self, path: str, export_single_mesh: bool = True):
        trimesh.exchange.export.export_mesh(
            self.tsdf_fuser_pred.tsdf.to_mesh(export_single_mesh=export_single_mesh),
            path,
        )

    def get_mesh(self, export_single_mesh: bool = True) -> trimesh.Trimesh:
        return self.tsdf_fuser_pred.tsdf.to_mesh(export_single_mesh=export_single_mesh)

    def get_tsdf(self):
        return self.tsdf_fuser_pred.tsdf


# __AVAILABLE_FUSERS__ = {"ours": OurFuser, "open3d": Open3DFuser}


# def get_fuser(opts, scan: Optional[str] = None) -> DepthFuser:
#     """Returns the depth fuser required"""

#     if opts.inference.depth_fuser not in __AVAILABLE_FUSERS__.keys():
#         raise ValueError(
#             f"Selected TSDF fuser {opts.depth_fuser} not found. Available fusers are {__AVAILABLE_FUSERS__.keys()}"
#         )
#     gt_path = None
#     if opts.data.dataset == "scannet":
#         gt_path = ScannetDataset.get_gt_mesh_path(opts.data.dataset_path, opts.split, scan)

#     return __AVAILABLE_FUSERS__[opts.inference.depth_fuser](
#         gt_path=gt_path,
#         fusion_resolution=opts.inference.fusion_resolution,
#         max_fusion_depth=opts.inference.fusion_max_depth,
#         fuse_color=opts.inference.fuse_color,
#     )
