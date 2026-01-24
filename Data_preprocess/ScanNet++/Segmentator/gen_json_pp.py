# 2025-03-10 用于将segmentor输出的 segIndices 数据 转为 segments 文件
import os
import json
import argparse
import numpy as np
from tqdm import trange

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, default='scene1111_00')
    parser.add_argument('--path_test', type=str, help='path to scans_test folder')
    args = parser.parse_args()
    scene_id = args.scene_id
    path_test = args.path_test

    os.rename(f'{path_test}/{scene_id}/mesh_aligned_0.05.ply', \
    f'{path_test}/{scene_id}/{scene_id}_vh_clean_2.ply')
    os.rename(f'{path_test}/{scene_id}/mesh_aligned_0.05_semantic.ply', \
    f'{path_test}/{scene_id}/{scene_id}_vh_clean_2.labels.ply')
    os.rename(f'{path_test}/{scene_id}/mesh_aligned_0.05.0.010000.segs.json', \
    f'{path_test}/{scene_id}/{scene_id}_vh_clean_2.0.010000.segs.json')
    
    depthHeight = 192
    depthWidth = 256
    content = f"depthHeight = {depthHeight}\ndepthWidth = {depthWidth}"
    with open(f'{path_test}/{scene_id}/{scene_id}.txt', 'w', encoding='utf-8') as file:
        file.write(content)

    indices_file = f'{path_test}/{scene_id}/{scene_id}_vh_clean_2.0.010000.segs.json'
    segments_file = f'{path_test}/{scene_id}/segments_anno.json'
    segments_out_file = f'{path_test}/{scene_id}/{scene_id}.aggregation_pp.json'
    segindices = np.array(json.load(open(indices_file, 'r'))['segIndices'])
    segments = json.load(open(segments_file, 'r'))
    seggroupes = segments['segGroups']
    outgroupes = []
    print(len(seggroupes))
    for i in trange(len(seggroupes)):
        object_i = {}
        object_i['id'] = seggroupes[i]['id']
        object_i['objectId'] = seggroupes[i]['objectId']
        object_i['label'] = seggroupes[i]['label']
        seg_segments = np.array(seggroupes[i]['segments'])
        out_segments = np.unique(segindices[seg_segments])
        object_i['segments'] = out_segments.tolist()
        outgroupes.append(object_i)

    segments_out = {}
    segments_out['sceneId'] = segments['sceneId']
    segments_out['appId'] = segments['appId']
    segments_out['segGroups'] = outgroupes
    json.dump(segments_out, open(segments_out_file, 'w'))
