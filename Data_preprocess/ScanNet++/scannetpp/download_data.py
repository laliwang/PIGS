# 2025-03-17 直接从对应网址上下载所需的一系列文件
# rgb.mkv, depth.bin, pose_intrinsic_imu.json, segments.json, segments_anno.json, mesh_aligned_0.05.ply, mesh_aligned_0.05_semantic.ply
import os
import sys
import subprocess
import argparse

def download_file(url, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    file_name = url.split('/')[-1]
    command = ['wget', url, '-O', os.path.join(output_directory, file_name)]
    # command = ['wget', url, '-P', output_directory, '--content-disposition']
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"{url.split('/')[-1]} download failed!: {e}")

parser = argparse.ArgumentParser(description='Process scene name.')
parser.add_argument('--scene', type=str, required=True, help='Name of the scene to process.')
parser.add_argument('--token', type=str, required=True, help='token')
parser.add_argument('--output_dir', type=str, required=True, help='Name of the output directory.')
args = parser.parse_args()
scene = args.scene
token = args.token
output_dir = args.output_dir

iphone_path = f'https://scannetpp.mlsg.cit.tum.de/scannetpp/download/v2?version=v1&token={token}&file=data/{scene}/iphone'
scans_path = f'https://scannetpp.mlsg.cit.tum.de/scannetpp/download/v2?version=v1&token={token}&file=data/{scene}/scans'

tgt_iphone_path = f'{output_dir}/{scene}/iphone'
tgt_scans_path = f'{output_dir}/{scene}/scans'
os.makedirs(tgt_iphone_path, exist_ok=True)
os.makedirs(tgt_scans_path, exist_ok=True)
iphone_list = ['rgb.mkv', 'depth.zip', 'pose_intrinsic_imu.zip', 'colmap.zip']
scans_list = ['segments.zip', 'segments_anno.zip', 'mesh_aligned_0.05.zip', 'mesh_aligned_0.05_semantic.zip']

for file in iphone_list:
    download_file(f'{iphone_path}/{file}', tgt_iphone_path)
    if file.endswith('.zip'):
        subprocess.run(['unzip', '-o', os.path.join(tgt_iphone_path, file), '-d', tgt_iphone_path])
        os.remove(os.path.join(tgt_iphone_path, file))
for file in scans_list:
    download_file(f'{scans_path}/{file}', tgt_scans_path)
    if file.endswith('.zip'):
        subprocess.run(['unzip', '-o', os.path.join(tgt_scans_path, file), '-d', tgt_scans_path])
        os.remove(os.path.join(tgt_scans_path,file))