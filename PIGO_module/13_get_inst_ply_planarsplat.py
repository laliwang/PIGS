import os
import argparse
import numpy as np
import trimesh
import open3d as o3d
from tqdm import tqdm

def color_key(c, scale=255):
    """Map float color to a stable integer tuple key."""
    return tuple((c * scale).round().astype(int))

def sample_points_from_faces(vertices, faces, face_normals, num_samples_per_face=3):
    """
    Uniformly sample a fixed number of points from each triangular face.
    Algorithm: P = (1 - sqrt(r1)) * A + (sqrt(r1) * (1 - r2)) * B + (sqrt(r1) * r2) * C
    """
    num_faces = faces.shape[0]
    
    # Extract triangle vertices A, B, C [num_faces, 3]
    A = vertices[faces[:, 0]]
    B = vertices[faces[:, 1]]
    C = vertices[faces[:, 2]]
    
    # Generate random barycentric coordinates [num_faces, num_samples, 1]
    r1 = np.random.rand(num_faces, num_samples_per_face, 1).astype(np.float32)
    r2 = np.random.rand(num_faces, num_samples_per_face, 1).astype(np.float32)
    
    sqrt_r1 = np.sqrt(r1)
    wA = 1.0 - sqrt_r1
    wB = sqrt_r1 * (1.0 - r2)
    wC = sqrt_r1 * r2
    
    # Calculate sampled point coordinates [num_faces, num_samples, 3]
    # Using broadcasting: (F, S, 1) * (F, 1, 3) -> (F, S, 3)
    sampled_pts = (wA * A[:, None, :]) + (wB * B[:, None, :]) + (wC * C[:, None, :])
    
    # Inherit face normals: each sampled point shares the normal of its parent face
    # [num_faces, 3] -> [num_faces, num_samples, 3]
    sampled_normals = np.tile(face_normals[:, None, :], (1, num_samples_per_face, 1))
    
    return sampled_pts, sampled_normals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample plane point clouds from color-labeled Mesh")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to input .obj or .ply mesh")
    parser.add_argument("--color_path", type=str, required=True, help="Path to input color list .npy")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--samples_per_face", type=int, default=5, help="Number of points to sample per face")
    args = parser.parse_args()

    # ==================== Path Handling ====================
    mesh_path = args.mesh_path
    color_path = args.color_path

    if args.out_dir is None:
        # Default output to mesh_dir/../mesh/plane_pcd
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(mesh_path)),
            "mesh/plane_pcd"
        )
    else:
        out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # ==================== Load Mesh ====================
    # process=False ensures vertex order and face structure remain unchanged
    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)

    if mesh.visual.face_colors is None:
        raise RuntimeError("Mesh has no face_colors; cannot determine plane IDs")

    # Get face colors and convert to 0-1 float
    face_colors_raw = mesh.visual.face_colors[:, :3].astype(np.float32) / 255.0

    # ==================== Load Instance Color Map ====================
    inst_color_list = np.load(color_path).astype(np.float32)
    # Create Color -> ID mapping
    color2id = {color_key(c): i for i, c in enumerate(inst_color_list)}

    # ==================== Face Classification (Face -> Plane ID) ====================
    face_ids = np.array([color2id.get(color_key(c), 0) for c in face_colors_raw], dtype=np.int32)
    unique_ids = np.unique(face_ids)
    print(f"[INFO] {len(unique_ids) - 1} planes detected (excluding ID 0)")

    # ==================== Execute Uniform Sampling ====================
    print(f"[INFO] Sampling from {len(faces)} faces ({args.samples_per_face} pts/face)...")
    all_sampled_pts, all_sampled_nors = sample_points_from_faces(
        vertices, faces, face_normals, args.samples_per_face
    )

    # ==================== Export Point Clouds by ID ====================
    for pid in tqdm(unique_ids, desc="Exporting plane point clouds"):
        if pid == 0:
            print(f"[INFO] Skipping ID 0 (background/unlabeled)")
            continue

        # Find all face indices belonging to the current plane
        face_mask = (face_ids == pid)
        
        # Extract and flatten sampled points and normals
        # shape: [num_mask_faces * samples_per_face, 3]
        pts = all_sampled_pts[face_mask].reshape(-1, 3)
        nors = all_sampled_nors[face_mask].reshape(-1, 3)
        
        # Set uniform color from the instance color list
        cols = np.tile(inst_color_list[pid], (len(pts), 1))

        # Build Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.normals = o3d.utility.Vector3dVector(nors)
        pcd.colors = o3d.utility.Vector3dVector(cols)

        out_path = os.path.join(out_dir, f"plane_{pid:04d}.ply")
        o3d.io.write_point_cloud(out_path, pcd)

    print(f"[DONE] All plane point clouds saved to: {out_dir}")