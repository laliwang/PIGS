import open3d as o3d
import numpy as np
import hdbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import trimesh
from argparse import ArgumentParser, Namespace
import sys
import os

def save_mesh_to_ply(mesh: trimesh.Trimesh, output_path):
    """
    保存网格到PLY文件。
    
    参数：
        mesh (trimesh.Trimesh): 输入的三维网格
        output_path (Path): 输出的PLY文件路径
    """
    # 确保mesh是Trimesh类型
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("传入的 mesh 对象无效，必须是 trimesh.Trimesh 类型。")
    
    result = trimesh.exchange.ply.export_ply(mesh, encoding="ascii")
    with open(output_path, "wb+") as fh:
        fh.write(result)
        fh.close()


def process_point_cloud_by_color(input_file, output_dir, radius=0.1):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    # 读取点云
    pcd = o3d.io.read_point_cloud(input_file)
    
    # 获取点云的颜色信息和法向量
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)

    # 用一个字典来存储颜色相同的点
    color_clusters = {}
    ids=0
    # 遍历每个点，将其颜色作为键，点的索引作为值存入字典
    for i, color in enumerate(colors):
        color_key = tuple(np.round(color, 3))  # 使用 rounded color 作为键（防止浮动误差）
        if color_key not in color_clusters:
            color_clusters[color_key] = []
        color_clusters[color_key].append(i)

    # 对每个簇计算法向量的平均值，并更新簇内点的法向量
    combined_mesh=o3d.geometry.TriangleMesh()
    for cluster_points_indices in color_clusters.values():
        # 获取该簇的法向量
        cluster_points = points[cluster_points_indices]
        cluster_normals = normals[cluster_points_indices]
        average_normal = np.mean(cluster_normals, axis=0)
        average_normal=average_normal / np.linalg.norm(average_normal)  # 单位化法向量
        cluster_normals=np.tile(average_normal, (len(cluster_points), 1)) 
        cluster_colors = colors[cluster_points_indices]
        distances = np.dot(cluster_points, average_normal)
        mean_distance = np.mean(distances)    
        cluster_points= cluster_points - np.outer(distances - mean_distance, average_normal)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster_points)  
        pcd.colors = o3d.utility.Vector3dVector(cluster_colors)  
        pcd.normals = o3d.utility.Vector3dVector(cluster_normals)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius]))
        if mesh.is_empty():
            print(f"输入的网格{i}为空，跳过处理。")
            continue

        combined_mesh+=mesh
        ids+=1
    output_file = output_dir
    o3d.io.write_triangle_mesh(output_file, combined_mesh)
    print(f"Combined mesh saved as: {output_file}")
    return ids



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--input_pcd_path', type=str, default="127.0.0.1")
    parser.add_argument('--output_mesh_path', type=str, default="127.0.0.1")
    parser.add_argument('--bpa_radius', type=float, default=0.1, help='Ball pivoting radius')
    args = parser.parse_args(sys.argv[1:])

    print("Pipeline Started...")
    num_id=process_point_cloud_by_color(args.input_pcd_path,args.output_mesh_path, radius=args.bpa_radius)
    print(f"Combined mesh ids:",num_id)