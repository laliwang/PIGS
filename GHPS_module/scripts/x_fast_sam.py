import argparse
import os
import sys
import cv2
sys.path.append('/code1/wjh/VLM_model/FastSAM')

from fastsam import FastSAM, FastSAMPrompt 
import torch
from PIL import Image
import numpy as np
from tqdm import trange
from natsort import natsorted
from utils.tools import convert_box_xywh_to_xyxy

def get_fast_sam(weight_folder):
    sam_model = FastSAM(os.path.join(weight_folder, 'FastSAM-x.pt'))
    return sam_model

def fast_predictor(sam_model, image, params):
    image_pil = Image.fromarray(image)
    everything_results = sam_model(
            image_pil,
            device=params['device'],
            retina_masks=params['retina'],
            imgsz=params['imgsz'],
            conf=params['conf'],
            iou=params['iou']    
        )
    prompt_process = FastSAMPrompt(image_pil, everything_results, device=params['device'])
    ann = prompt_process.everything_prompt()
    if ann == []:
        return None
    sam_mask = ann.cpu().numpy()
    return sam_mask


if __name__ == '__main__':

    image_folder = '/code3/wjh/2025-PIGS/TEST/debug/images'
    output_folder = '/code3/wjh/2025-PIGS/TEST/debug/output_fast'
    os.makedirs(output_folder, exist_ok=True)
    image_paths = natsorted([os.path.join(image_folder, path) for path in os.listdir(image_folder)])

    weight_folder = '/code3/wjh/2025-PIGS/PIGS/weights'
    sam_model = get_fast_sam(weight_folder)
    params = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'retina': True,
        'imgsz': 1024,
        'conf': 0.4,
        'iou': 0.9
    }


    for i in trange(len(image_paths)):
        image_path = image_paths[i]
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sam_mask = fast_predictor(sam_model, image, params)
        print(sam_mask.shape)

