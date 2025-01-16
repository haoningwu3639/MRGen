import json
import os
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
import shutil
import random
from torchvision import transforms
from einops import repeat

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

COLORS = [
    [0, 0, 0],
    [0, 0, 255],     # 红色
    [0, 255, 0],     # 绿色
    [255, 0, 0],     # 蓝色
    [0, 255, 255],   # 黄色
    [255, 0, 255],   # 紫色
    [255, 255, 0],   # 青色
    [128, 0, 0],     # 暗红色
    [0, 128, 0],     # 暗绿色
    [0, 0, 128],     # 暗蓝色
    [128, 128, 0],   # 橄榄色
    [128, 0, 128],   # 深紫色
    [0, 128, 128],   # 青绿
    [192, 192, 192], # 银色
    [128, 128, 128], # 灰色
    [0, 0, 0],       # 黑色
    [255, 255, 255], # 白色
    [255, 165, 0],   # 橙色
    [75, 0, 130],    # 靛青色
    [238, 130, 238], # 粉紫色
    [173, 216, 230], # 浅蓝色
]

def resize_with_padding(slice, target_size=512, nearest_mode=True):
    # 获取原始数组的高度和宽度
    h, w = slice.shape
    
    # 计算需要填充的尺寸
    if h > w:
        pad_width = (h - w) // 2
        padding = ((0, 0), (pad_width, h - w - pad_width))
    else:
        pad_height = (w - h) // 2
        padding = ((pad_height, w - h - pad_height), (0, 0))
    
    # 使用 np.pad 进行填充
    padded = np.pad(slice, padding, mode='constant', constant_values=0)
    
    # 确保填充后的尺寸是方形
    assert padded.shape[0] == padded.shape[1], "Padding did not result in a square matrix."
    
    # 使用 OpenCV 进行缩放（插值方式可根据需求更改）
    if nearest_mode:
        resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    else:
        resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return resized

def prepare_nnUNet_raw_png(
    json_file, 
    mask2label, 
    dataset_id, 
    dataset_name, 
    raw_data_dir='your path to nnUNet raw data dir'):
    
    mask2label = MASK2LABEL[mask2label]

    modality = {0:"MR"}
    file_ending = '.png'

    with open(json_file, 'r') as f:
        data = json.load(f)
        
    raw_data_dir = os.path.join(raw_data_dir, f'Dataset{dataset_id}_'+dataset_name)
    Path(os.path.join(raw_data_dir, 'imagesTr')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(raw_data_dir, 'labelsTr')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(raw_data_dir, 'visualizationTr')).mkdir(parents=True, exist_ok=True)
    
    labels = {
        "background": 0
    }
    cursor = 1
    for mask_path_key, label_name in mask2label.items():
        labels[label_name] = cursor
        cursor += 1
    
    num_train = 0
    no_anno_count = 0
    for datum in tqdm(data):
        
        # {
        #     "prompt": [
        #         "T2 SPIR Abdomen MRI, fat low signal, muscle low signal, water high signal, fat dark, water bright, Upper Abdominal Region, liver, right kidney, left kidney, spleen, kidney"
        #     ],
        #     "modality": "T1 in-phase Abdomen MRI",
        #     "region": "Upper Abdominal Region",
        #     "label": "liver, right kidney, left kidney, spleen, kidney",
        #     "mask_path": "../synthetic_mask/0_mask.png",
        #     "mask_0_path": "../synthetic_mask/0_mask_0.png",    # liver
        #     "mask_1_path": "../synthetic_mask/0_mask_1.png",    # right kidney
        #     "mask_2_path": "../synthetic_mask/0_mask_2.png",    # left kidney
        #     "mask_3_path": "../synthetic_mask/0_mask_3.png",    # spleen
        #     "image_path": "../synthetic_T2-SPIR/0_T2-SPIR Abdomen MRI_0.png",
        #     "combined_path": "../synthetic_T2-SPIR_combined/0_T2-SPIR Abdomen MRI_0_combined.png"
        # },
        
        # load image
        image_path = datum['image_path']
        assert os.path.exists(image_path), f"{image_path} not exists"
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # merge masks into a sc mask
        color_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        sc_mask = []
        for mask_path_key, label_name in mask2label.items():
            source_path = datum[mask_path_key]
            assert os.path.exists(source_path), f"{source_path} not exists"
            tmp = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
            tmp = np.where(tmp>0, 1, 0)
            assert tmp.max() <= 1 and tmp.min() == 0, f"{source_path} out of boundary: max={tmp.max()}, min={tmp.min()}"
            sc_mask.append(labels[label_name] * tmp)
            color_mask[tmp == 1] = COLORS[labels[label_name]]
        sc_mask = np.stack(sc_mask, axis=0) # 4, 512, 512
        sc_mask = np.max(sc_mask, axis=0)
        
        if np.all(sc_mask == 0):
            print(f"The sc_mask for image {image_path} is all zeros.")
            no_anno_count += 1
            
        # save
        image_name = image_path.split('/')[-1].split('.')[0]
        cv2.imwrite(os.path.join(raw_data_dir, "imagesTr", image_name+'_0000.png'), image)
        cv2.imwrite(os.path.join(raw_data_dir, "labelsTr", image_name+'.png'), sc_mask)
        
        # visualization
        color_image = repeat(image, 'h w -> h w c', c=3)
        color_image = (color_image - color_image.min()) / (color_image.max() - color_image.min()) * 255.0
        color_image = color_image * 0.7 + color_mask * 0.3
        cv2.imwrite(os.path.join(raw_data_dir, "visualizationTr", image_name+'.png'), color_image)
        
        num_train += 1
        
    generate_dataset_json(
        raw_data_dir, 
        channel_names=modality,
        labels=labels,
        # regions_class_order=(1, 3, 2),
        num_training_cases=num_train, 
        file_ending=file_ending,
        dataset_name=dataset_name, 
        reference='none',
        # overwrite_image_reader_writer=image_reader_writer,
        description=dataset_name)
    
    print(f'Create Dataset{dataset_id}_{dataset_name} under nnUNet_raw, {num_train} training samples.')
    print(f'WARNING: {no_anno_count} out of {num_train} samples are empty (no annotation) !')
    
def merg_json(json_ls, ratio_ls, save_path):
    data = []
    for json_file, ratio in zip(json_ls, ratio_ls):
        sample_num = 0
        
        with open(json_file, 'r') as f:
            tmp = json.load(f)
            
        for datum in tmp:
            if random.random() < ratio:
                data.append(datum)
                sample_num += 1
                
        print(f'{json_file} sample num: {sample_num}')
            
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
        
if __name__ == '__main__':
    
    MASK2LABEL = {
        'CHAOSMR': {
        'mask_0_path':'liver',
        'mask_1_path':'right kidney',
        'mask_2_path':'left kidney',
        'mask_3_path':'spleen',
        },
        'Prostate': {
            'mask_0_path':'prostate'
        },
        'Liver': {
            'mask_0_path':'liver'
        },
        'Liver Tumor': {
            'mask_0_path':'liver',
            'mask_1_path':'liver tumor'
        },
        'Pancreas': {
            'mask_0_path':'pancreas'
        },
    }
    
    import argparse
    parser = argparse.ArgumentParser(description='Transfer the synthetic data to nnUNet raw data format.')
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--mask2label', type=str, required=True, default='CHAOSMR')
    parser.add_argument('--dataset_id', type=str, required=True, default='001', help='a three digital number from 001 to 999')
    parser.add_argument('--dataset_name', type=str, required=True, default='CHAOSMR')
    
    args = parser.parse_args()
    
    prepare_nnUNet_raw_png(args.json_file, None, args.mask2label, args.dataset_id, args.dataset_name)
    
    
    
    