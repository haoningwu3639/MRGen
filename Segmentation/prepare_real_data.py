import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import json
from torchvision import transforms
import monai
import torch
from einops import repeat, rearrange, reduce

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

class Loader_Wrapper():
    """
    different from SAT format, when trans to nnUNET 2D format (png):
    1. no spacing and no crop
    2. no merged labels and masks
    3. no intensity normalization
    """
    
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def CHAOS_MRI(self, datum:dict) -> tuple:
        """
        'liver', 
        'right kidney', 
        'left kidney', 
        'spleen'
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label'][:4]
        
        # NOTE: merge label
        # kidney = mask[1] + mask[2]
        # mask = torch.cat((mask, kidney.unsqueeze(0)), dim=0)
        # labels.append("kidney")
        
        #img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = (mask-mask.min())/(mask.max()-mask.min() + 1e-10) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def PROMISE12(self, datum:dict) -> tuple:
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image']
        mask = dictionary['label']
        
        labels = datum['label']
        
        #img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def AMOS22_CT(self, datum:dict) -> tuple:
        """
        labels = [
            'spleen', 
            'right kidney',
            'left kidney',
            'gallbladder',
            'esophagus',
            'liver',
            'stomach',
            'aorta',
            'inferior vena cava',
            'pancreas',
            'right adrenal gland',
            'left adrenal gland',
            'duodenum',
            'urinary bladder',
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:14]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        # merge label
        # mc_masks.append(mc_masks[1]+mc_masks[2])
        # labels.append("kidney")
        # mc_masks.append(mc_masks[10]+mc_masks[11])
        # labels.append("adrenal gland")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'CT')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def AMOS22_MRI(self, datum:dict) -> tuple:
        """
        labels = [
            'spleen', 
            'right kidney',
            'left kidney',
            'gallbladder',
            'esophagus',
            'liver',
            'stomach',
            'aorta',
            'inferior vena cava',
            'pancreas',
            'right adrenal gland',
            'left adrenal gland',
            'duodenum',
            'urinary bladder',
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        labels = datum['label'][:14]
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        
        # merge label
        # mc_masks.append(mc_masks[1]+mc_masks[2])
        # labels.append("kidney")
        # mc_masks.append(mc_masks[10]+mc_masks[11])
        # labels.append("adrenal gland")
        
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'MRI')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def ATLAS(self, datum:dict) -> tuple:
        """
        labels = [
            "liver",
            "liver tumor",
            ]
            
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D 
        
        mc_masks = []
        labels = datum['label'][:2]
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
        mc_masks[0] += mc_masks[1]

        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'MRI' )

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def PanSeg(self, datum:dict) -> tuple:
        """
        labels = [
            'pancreas',
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        # original
        labels = datum['label'][:1]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'MRI')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def LiQA(self, datum:dict) -> tuple:
        """
        labels = [
            'liver',
            ]
        """
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [1, H, W, D]
        
        # original
        labels = datum['label'][:1]
        
        ##img = repeat(img, 'c h w d -> (c r) h w d', r=3)
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'MRI')

        return img, mask, labels, datum['modality'], datum['image'], datum['mask']

    def MSD_Prostate(self, datum:dict) -> tuple:
        mod2channel = {"T2":0, "ADC":1}
        tmp = datum['image'].split('/')
        mod = tmp[-1]
        channel = mod2channel[mod]
        img_path = '/'.join(tmp[:-1])
        
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label'], ensure_channel_first=True),
                #monai.transforms.AddChanneld(keys=['image']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':img_path, 'label':datum['mask']})
        img = dictionary['image'][channel, :, :, :] # [H, W, D]
        mask = dictionary['label'] # [1, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)
            
        # mc_masks.append(mc_masks[0]+mc_masks[1]) 
        # labels.append('prostate')
        
        mask = torch.cat(mc_masks, dim=0) # [3, H, W, D]
        
        mask = (mask-mask.min())/(mask.max()-mask.min()) # normal到0~1之间
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        
        img = repeat(img, 'h w d -> c h w d', c=1)  # [C, H, W, D]
        # img = Normalization(img, 'MRI')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    
    def MSD_Liver(self, datum:dict) -> tuple:
        monai_loader = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=['image', 'label']),
                monai.transforms.AddChanneld(keys=['image', 'label']),
                monai.transforms.Orientationd(axcodes="RAS", keys=['image', 'label']),
                # monai.transforms.Spacingd(keys=["image", "label"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
        dictionary = monai_loader({'image':datum['image'], 'label':datum['mask']})
        img = dictionary['image'] # [1, H, W, D]
        mask = dictionary['label']# [C, H, W, D]
        
        labels = datum['label'][:2]
        
        mc_masks = []
        for i, label in enumerate(labels):
            binary_mask = torch.where(mask==(i+1), 1.0, 0.0)    # 0 is background, so plus one is the correct label
            mc_masks.append(binary_mask)    # [1, H, W, D]
        mask = torch.concat(mc_masks, dim=0) # [C, H, W, D]
        # 0 is liver, 1 is liver tumor, should be included in liver
        # mask[0] += mask[1]
        
        mask = torch.where(mask>0.5, 1.0, 0.0)    # 去除mask上的噪声，强制0或1
        # img = Normalization(img, 'CT')
        
        return img, mask, labels, datum['modality'], datum['image'], datum['mask']
    

class Name_Mapper():
    """
    Rule to name 2D png files for each dataset
    """
    
    def __init__(self):
        pass

    def CHAOS_MRI(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-5]+'_'+img_path.split('/')[-4]+'_'+img_path.split('/')[-3]+'_'+img_path.split('/')[-2]+'_'+img_path.split('/')[-1]
        target_img_filename = target_img_filename.replace('.nii.gz', '')
        return target_img_filename
    
    def MSD_Prostate(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-2]+'_'+img_path.split('/')[-1]   # prostate_35.nii.gz_T2
        target_img_filename = target_img_filename.replace('.nii.gz', '')    # prostate_35_T2
        return target_img_filename
    
    def AMOS22_CT(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-1]   # amos_0033.nii.gz
        target_img_filename = target_img_filename.replace('.nii.gz', '')    # amos_0033
        return target_img_filename
    
    def MSD_Liver(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-1]   # liver_37.nii.gz
        target_img_filename = target_img_filename.replace('.nii.gz', '')    # liver_37
        return target_img_filename
    
    def PROMISE12(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-1]   # Case15.nii.gz
        target_img_filename = target_img_filename.replace('.nii.gz', '')    # Case15
        return target_img_filename
    
    def LiQA(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-3]+'_'+img_path.split('/')[-2]+'_'+img_path.split('/')[-1]   # Vendor_A_1920-A-S1_GED4.nii.gz
        target_img_filename = target_img_filename.replace('.nii.gz', '')    # Vendor_A_1920-A-S1_GED4
        return target_img_filename
    
    def ATLAS(self, img_path:str) -> str:
        target_img_filename = img_path.split('/')[-1]   # im52.nii.gz
        target_img_filename = target_img_filename.replace('.nii.gz', '')    # im52
        return target_img_filename
    
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

def prepare_nnunet_PanSeg_raw(t1_id, t2_id):
    """
    We have a special preprocess for PanSeg data
    """
    modality = {0:"MR"}
    file_ending = '.png'
    labels = {
        "background": 0,
        "pancreas": 1
    }
    
    Path(f'{nnUNet_RAW_PATH}/Dataset{t2_id}_PanSeg_T2/imagesTr').mkdir(parents=True, exist_ok=True)
    Path(f'{nnUNet_RAW_PATH}/Dataset{t2_id}_PanSeg_T2/labelsTr').mkdir(parents=True, exist_ok=True)
    num_train = 0
    with open('Segmentation/data/trainsets/PanSeg_T2.json', 'r') as f:
        train_data = json.load(f)
    for datum_dict in train_data:
        image_path = datum_dict['image_path']   # xxxx/AHN_0004_0000_0.png
        volume_id, slice_id = image_path.split('/')[-1].split('_0000_')
        slice_id = slice_id.replace('.png', '')
        # move image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = resize_with_padding(image, target_size=512, nearest_mode=False)
        assert image.shape == (512, 512), f'image shape {image.shape}'  
        volume_id = 't2_'+volume_id
        new_file_name = f'{volume_id}_s{slice_id}_0000.png'   # t2_MCF_0001_s16_0000.png
        cv2.imwrite(f'{nnUNet_RAW_PATH}/Dataset{t2_id}_PanSeg_T2/imagesTr/{new_file_name}', image)
        # move mask
        mask_path = datum_dict['mask0_path']   # xxxx/AHN_0004_0000_0.png
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = resize_with_padding(mask, target_size=512, nearest_mode=True)
        mask = np.where(mask>0, 1, 0)
        assert mask.shape == (512, 512), f'mask shape {mask.shape}'
        cv2.imwrite(f'{nnUNet_RAW_PATH}/Dataset{t2_id}_PanSeg_T2/labelsTr/{new_file_name.replace("_0000.png", ".png")}', mask)
        num_train += 1
        
    Path(f'{nnUNet_RAW_PATH}/Dataset{t2_id}_PanSeg_T2/imagesTs').mkdir(parents=True, exist_ok=True)
    Path(f'{nnUNet_RAW_PATH}/Dataset{t2_id}_PanSeg_T2/labelsTs').mkdir(parents=True, exist_ok=True)
    with open('Segmentation/data/testsets/PanSeg_T2.json', 'r') as f:
        test_data = json.load(f)
    for datum_dict in test_data:
        image_path = datum_dict['image_path']   # xxxx/AHN_0004_0000_0.png
        volume_id, slice_id = image_path.split('/')[-1].split('_0000_')
        slice_id = slice_id.replace('.png', '')
        # move image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = resize_with_padding(image, target_size=512, nearest_mode=False)
        assert image.shape == (512, 512), f'image shape {image.shape}'  
        volume_id = 't2_'+volume_id
        new_file_name = f'{volume_id}_s{slice_id}_0000.png'   # t2_MCF_0001_s16_0000.png
        cv2.imwrite(f'{nnUNet_RAW_PATH}/Dataset{t2_id}_PanSeg_T2/imagesTs/{new_file_name}', image)
        # move mask
        mask_path = datum_dict['mask0_path']   # xxxx/AHN_0004_0000_0.png
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = resize_with_padding(mask, target_size=512, nearest_mode=True)
        mask = np.where(mask>0, 1, 0)
        assert mask.shape == (512, 512), f'mask shape {mask.shape}'
        cv2.imwrite(f'{nnUNet_RAW_PATH}/Dataset{t2_id}_PanSeg_T2/labelsTs/{new_file_name.replace("_0000.png", ".png")}', mask)
        
    generate_dataset_json(
        f'{nnUNet_RAW_PATH}/Dataset{t2_id}_PanSeg_T2', 
        channel_names=modality,
        labels=labels,
        num_training_cases=num_train, 
        file_ending=file_ending,
        dataset_name='PanSeg_T2', 
        reference='none',
        description='PanSeg_T2')
    
    print(f'Create Dataset{t2_id}_PanSeg_T2 under nnUNet_raw, {num_train} training samples')
 
    Path(f'{nnUNet_RAW_PATH}/Dataset{t1_id}_PanSeg_T1/imagesTr').mkdir(parents=True, exist_ok=True)
    Path(f'{nnUNet_RAW_PATH}/Dataset{t1_id}_PanSeg_T1/labelsTr').mkdir(parents=True, exist_ok=True)
    num_train = 0
    with open('Segmentation/data/trainsets/PanSeg_T1.json', 'r') as f:
        train_data = json.load(f)
    for datum_dict in train_data:
        image_path = datum_dict['image_path']   # xxxx/AHN_0004_0000_0.png
        volume_id, slice_id = image_path.split('/')[-1].split('_0000_')
        slice_id = slice_id.replace('.png', '')
        # move image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = resize_with_padding(image, target_size=512, nearest_mode=False)
        assert image.shape == (512, 512), f'image shape {image.shape}'  
        volume_id = 't1_'+volume_id
        new_file_name = f'{volume_id}_s{slice_id}_0000.png'   # t1_MCF_0001_s16_0000.png
        cv2.imwrite(f'{nnUNet_RAW_PATH}/Dataset{t1_id}_PanSeg_T1/imagesTr/{new_file_name}', image)
        # move mask
        mask_path = datum_dict['mask0_path']   # xxxx/AHN_0004_0000_0.png
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = resize_with_padding(mask, target_size=512, nearest_mode=True)
        mask = np.where(mask>0, 1, 0)
        assert mask.shape == (512, 512), f'mask shape {mask.shape}'
        cv2.imwrite(f'{nnUNet_RAW_PATH}/Dataset{t1_id}_PanSeg_T1/labelsTr/{new_file_name.replace("_0000.png", ".png")}', mask)
        
        num_train += 1
        
    Path(f'{nnUNet_RAW_PATH}/Dataset{t1_id}_PanSeg_T1/imagesTs').mkdir(parents=True, exist_ok=True)
    Path(f'{nnUNet_RAW_PATH}/Dataset{t1_id}_PanSeg_T1/labelsTs').mkdir(parents=True, exist_ok=True)
    with open('Segmentation/data/testsets/PanSeg_T1.json', 'r') as f:
        test_data = json.load(f)
    for datum_dict in test_data:
        image_path = datum_dict['image_path']   # xxxx/AHN_0004_0000_0.png
        volume_id, slice_id = image_path.split('/')[-1].split('_0000_')
        slice_id = slice_id.replace('.png', '')
        # move image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = resize_with_padding(image, target_size=512, nearest_mode=False)
        assert image.shape == (512, 512), f'image shape {image.shape}'  
        volume_id = 't1_'+volume_id
        new_file_name = f'{volume_id}_s{slice_id}_0000.png'   # t1_MCF_0001_s16_0000.png
        cv2.imwrite(f'{nnUNet_RAW_PATH}/Dataset{t1_id}_PanSeg_T1/imagesTs/{new_file_name}', image)
        # move mask
        mask_path = datum_dict['mask0_path']   # xxxx/AHN_0004_0000_0.png
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = resize_with_padding(mask, target_size=512, nearest_mode=True)
        mask = np.where(mask>0, 1, 0)
        assert mask.shape == (512, 512), f'mask shape {mask.shape}'
        cv2.imwrite(f'{nnUNet_RAW_PATH}/Dataset{t1_id}_PanSeg_T1/labelsTs/{new_file_name.replace("_0000.png", ".png")}', mask)
        
    generate_dataset_json(
        f'{nnUNet_RAW_PATH}/Dataset{t1_id}_PanSeg_T1', 
        channel_names=modality,
        labels=labels,
        num_training_cases=num_train, 
        file_ending=file_ending,
        dataset_name='PanSeg_T1', 
        reference='none',
        description='PanSeg_T1')
    
    print(f'Create Dataset{t1_id}_PanSeg_T1 under nnUNet_raw, {num_train} training samples')

def prepare_nnUNet_raw(train_jsonl, test_jsonl, nnunet_dataset_name, rescale=True, reorient=True):
    """
    convert train and test data (jsonl files in SAT format) into nnUNET raw data (png format):
    0. aligned orientation (based on SAT data_loader_v4)
    1. sc mask
    2. image normalized to 0~255
    3. save as grey_scale image
    """
    
    loader = Loader_Wrapper()
    name_mapper = Name_Mapper()
    
    Path(f'{nnUNet_RAW_PATH}/{nnunet_dataset_name}/imagesTr').mkdir(parents=True, exist_ok=True)
    Path(f'{nnUNet_RAW_PATH}/{nnunet_dataset_name}/labelsTr').mkdir(parents=True, exist_ok=True)
    Path(f'{nnUNet_RAW_PATH}/{nnunet_dataset_name}/imagesTs').mkdir(parents=True, exist_ok=True)
    Path(f'{nnUNet_RAW_PATH}/{nnunet_dataset_name}/labelsTs').mkdir(parents=True, exist_ok=True)
    
    num_training_cases = 0
    
    for jsonl, nnunet_split in zip([train_jsonl, test_jsonl], ['Tr', 'Ts']):
        if jsonl is None:
            continue
        
        with open(jsonl, 'r') as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]

        for sample in tqdm(data, desc=f'processing data in {jsonl}'):
            # load 3D data
            func_name = sample['dataset']
            batch = getattr(loader, func_name)(sample)
            # (1, H, W, D), (N, H, W, D)
            img_tensor, mc_mask, text_ls, _, image_path, _ = batch
            
            img = img_tensor.numpy()
            mc_mask = mc_mask.numpy()
            image_name = getattr(name_mapper, func_name)(image_path)
            
            # counter-clock 90 degree and flip horizontally
            if reorient:
                img = np.rot90(img, axes=(1, 2))
                img = np.flip(img, axis=2)
                mc_mask = np.rot90(mc_mask, axes=(1, 2))
                mc_mask = np.flip(mc_mask, axis=2)
            
            if nnunet_split == 'Tr':
                num_training_cases += img.shape[-1]
            
            # get label info
            labels = {
                "background": 0
            }
            cursor = 1
            for i, text in enumerate(text_ls):
                labels[text] = cursor
                cursor += 1
                
            # convert mc mask into sc mask
            sc_mask = []
            for i, label in enumerate(text_ls):    # 0, kidney ...
                tmp = mc_mask[i, :, :, :] * (i+1)   # --> 1
                sc_mask.append(tmp)
            sc_mask = np.stack(sc_mask, axis=0)   # (N, H, W, D)
            sc_mask = np.max(sc_mask, axis=0)   # H W D
            
            # process and save each slice into png
            for slice_idx in range(img.shape[-1]):
                # normalize image slice
                image_slice = img[0, :, :, slice_idx]
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-10)
                image_slice = image_slice * 255.0
                assert image_slice.min() >= 0 and image_slice.max() <= 255.0, f'{image_slice.min()}, {image_slice.max()}'
                # rescale 
                if rescale:
                    image_slice = resize_with_padding(image_slice, target_size=512, nearest_mode=False)
                # save image slice
                image_slice = Image.fromarray(image_slice)
                image_slice = image_slice.convert('L')
                target_img_path = f'{nnUNet_RAW_PATH}/{nnunet_dataset_name}/images{nnunet_split}/{image_name}_s{slice_idx}_0000.png'
                image_slice.save(target_img_path)
                
                # convert mc to sc
                pixel_num = 0
                mask_slice = sc_mask[:, :, slice_idx]
                for label, intensity in labels.items():
                    tmp = np.where(mask_slice==intensity, 1, 0).sum()
                    pixel_num += tmp
                assert pixel_num == mask_slice.shape[0] * mask_slice.shape[1], f'{pixel_num} != {mask_slice.shape[0] * mask_slice.shape[1]}'
                # rescale 
                if rescale:
                    mask_slice = resize_with_padding(mask_slice, target_size=512, nearest_mode=True)
                # save mask slice
                target_msk_path = f'{nnUNet_RAW_PATH}/{nnunet_dataset_name}/labels{nnunet_split}/{image_name}_s{slice_idx}.png'
                cv2.imwrite(target_msk_path, mask_slice)
    
    generate_dataset_json(
        output_folder=f'{nnUNet_RAW_PATH}/{nnunet_dataset_name}', 
        channel_names={0:"MR"},
        labels=labels,
        # regions_class_order=(1, 3, 2),
        num_training_cases=num_training_cases, 
        file_ending='.png', 
        dataset_name='_'.join(nnunet_dataset_name.split('_')[1:]),  # Dataset001_xxx --> xxx
        reference='none',
        # overwrite_image_reader_writer=image_reader_writer,
        description=f'transformed from {train_jsonl} and {test_jsonl}')


if __name__ == '__main__':
    
    nnUNet_RAW_PATH = '/mnt/hwfile/medai/zhaoziheng/SAM/nnUNet_data/nnUNet_raw'
    
    prepare_nnUNet_raw(
        train_jsonl='Segmentation/data/trainsets/CHAOS_MRI/CHAOS_MRI_T2_SPIR.jsonl', 
        test_jsonl='Segmentation/data/testsets/CHAOS_MRI/CHAOS_MRI_T2_SPIR.jsonl', 
        nnunet_dataset_name='Dataset002_CHAOSMR_T2_Real',
        rescale=True,
        reorient=True
    )