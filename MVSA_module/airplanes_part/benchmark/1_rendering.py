import os

# do this before importing numpy! (doing it right up here in case numpy is dependency of e.g. json)
os.environ["MKL_NUM_THREADS"] = "4"  # noqa: E402
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # noqa: E402
os.environ["OMP_NUM_THREADS"] = "4"  # noqa: E402
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # noqa: E402
from pathlib import Path

import sys
import click
import numpy as np
import torch
import tqdm
import trimesh
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from torch.utils.data import DataLoader
from natsort import natsorted
import shutil

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from airplanes_part.utils.generic_utils import read_scannetv2_filename
from airplanes_part.utils.rendering_utils import interpolate_face_attributes_nearest


def run_frame(
    mesh: Meshes,
    cam_T_world_b44: torch.Tensor,
    K_b44: torch.Tensor,
    image_size: torch.Tensor,
    raster_settings: RasterizationSettings,
    frame_name: str,
    save_path: Path,
    render_depth: bool,
) -> None:
    """Render a frame.
    Params:
        mesh: mesh to render, as pytorch3d.structures.Meshes
        cam_T_world_b44: camera pose from world to camera, as (b,4,4) tensor
        image_size: size of the image to render, as a tensor with two elements (H,W)
        raster_setting: setting for the rasterizer
        frame_name: final name of the frame
        save_path: path to the folder in which we are going to save the frame
    """

    R = cam_T_world_b44[:, :3, :3]
    T = cam_T_world_b44[:, :3, 3]
    K = K_b44[:, :3, :3]
    cams = cameras_from_opencv_projection(R=R, tvec=T, camera_matrix=K, image_size=image_size)

    mesh = mesh.cuda()
    cams = cams.cuda()

    rasterizer = MeshRasterizer(
        cameras=cams,
        raster_settings=raster_settings,
    )

    _mesh = mesh.extend(len(cams))
    fragments = rasterizer(_mesh)

    # nearest sampling
    faces_packed = _mesh.faces_packed()
    verts_features_packed = _mesh.textures.verts_features_packed()
    faces_verts_features = verts_features_packed[faces_packed]
    texture_bhw14 = interpolate_face_attributes_nearest(
        fragments.pix_to_face, fragments.bary_coords, faces_verts_features
    )

    # bilinear sampling
    bilinear_texture_bhw14 = _mesh.textures.sample_textures(fragments, _mesh.faces_packed())
    rendered_depth_bhw = fragments.zbuf[..., 0]

    # we want nearest for RGB and bilinear for alpha - so combine
    texture_bhw14[..., 3] = bilinear_texture_bhw14[..., 3]
    plane_ids = texture_bhw14.cpu().numpy()[0, ..., 0, :] * 255
    rendered_depth = rendered_depth_bhw.cpu().numpy().squeeze()

    # save image with plane ids
    plane_ids = plane_ids.astype("uint8")
    plane_ids = Image.fromarray(plane_ids)
    plane_ids.save(save_path / f"{frame_name}_planes.png")

    # save rendered depth map
    if render_depth:
        np.save(save_path / f"{frame_name}_depth.npy", rendered_depth)


def run(
    data_dir: Path,
    planes_dir: Path,
    output_dir: Path,
    height: int,
    width: int,
    render_depth: bool,
) -> None:
    """Run the rendering over all the scenes defined in filename file.
    For each camera position provided by the dataloader for the given the scene we generate an image
    where each pixel contains the plane ID and the depth map.
    """
    # restructured by laliwang at 2026/01/13
    scene_name = os.path.basename(output_dir)
    print(f"Rendering for scene: {scene_name}")

    save_path = output_dir / "frames"
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    mesh = None

    try:
        mesh_trimesh = trimesh.exchange.load.load(planes_dir / scene_name / "mesh_with_planes.ply")
    except ValueError:
        logger.warning(
            f"Could not load mesh! Trying to load a manually saved version at path {planes_dir / scene_name / 'annotation' / 'mesh_with_planes2.ply'}"
        )
        mesh_trimesh = trimesh.exchange.load.load(planes_dir / scene_name / "mesh_with_planes2.ply")

    mesh = Meshes(
        verts=[torch.tensor(mesh_trimesh.vertices).float()],
        faces=[torch.tensor(mesh_trimesh.faces).float()],
        textures=TexturesVertex(
            torch.tensor(mesh_trimesh.visual.vertex_colors).unsqueeze(0).float() / 255.0
        ),
    )
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    image_size = torch.tensor((height, width)).unsqueeze(0)

    K_path = data_dir / scene_name / "intrinsic/intrinsic_depth.txt"
    pose_folder = data_dir / scene_name / "sensor_data"
    pose_paths = natsorted(list(pose_folder.glob("*.txt")))
    print(f"Processing {len(pose_paths)} poses")

    depth_path_0 = str(pose_paths[0]).replace("pose.txt", "depth.png")
    depth_0 = Image.open(depth_path_0)
    width_o, height_o = depth_0.size
    s_x = width / width_o
    s_y = height / height_o

    print(f"original image size: {width_o} x {height_o}")
    print(f"rendered image size: {width} x {height}")

    K_b44 = np.loadtxt(str(K_path))
    K_b44[0, 0] *= s_x
    K_b44[1, 1] *= s_y
    K_b44[0, 2] *= s_x
    K_b44[1, 2] *= s_y
    K_b44 = torch.from_numpy(K_b44).to(torch.float).unsqueeze(0)

    image_num = len(pose_paths)
    if image_num < 1000:
        step = 1
    elif image_num < 2000:
        step = 2
    elif image_num < 3000:
        step = 3
    elif image_num < 4000:
        step = 4
    else:
        step = 5
    print(f'Loaded {image_num} images, selected evenly sample step = {step}')

    for i, pose_path in tqdm.tqdm(enumerate(pose_paths[::step])):
        frame_name = os.path.basename(pose_path).split(".")[0].split("-")[-1]
        cam_T_world_b44 = torch.tensor(np.linalg.inv(np.loadtxt(str(pose_path))), dtype=torch.float).unsqueeze(0)
        
        run_frame(
                    mesh=mesh,
                    cam_T_world_b44=cam_T_world_b44,
                    K_b44=K_b44,
                    image_size=image_size,
                    raster_settings=raster_settings,
                    save_path=save_path,
                    frame_name=frame_name,
                    render_depth=render_depth,
                )

@click.command()
@click.option(
    "--data-dir",
    type=Path,
    help="Path to ScanNetv2",
)
@click.option(
    "--planes-dir",
    type=Path,
    help="Path to root folder generated by PlaneRCNN",
)
@click.option("--output-dir", type=Path, default="data/rendered_planes")
@click.option("--height", type=int, help="height of the image to render", default=192)
@click.option("--width", type=int, help="width of the image to render", default=256)
@click.option("--render-depth", is_flag=True, help="Render depth maps", default=False)
def cli(
    data_dir: Path,
    planes_dir: Path,
    output_dir: Path,
    height: int,
    width: int,
    render_depth: bool,
):
    torch.manual_seed(10)
    np.random.seed(10)

    run(
        data_dir=data_dir,
        planes_dir=planes_dir,
        output_dir=output_dir,
        height=height,
        width=width,
        render_depth=render_depth,
    )


if __name__ == "__main__":
    cli()  # type: ignore
