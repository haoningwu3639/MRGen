import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from utils.util import _process_image, _process_mask

# 3D-to-2D Pre-processing Rotation: CHAOS_MRI: 90; LiQA: -90 + flip; MSD_Prostate: 90; PanSeg: -90 + flip; PanSeg_MCF: 90 + flip; PROMISE: -90
class CHAOS_MRI_Dataset(Dataset): 
    def __init__(self, target_modality = 'T2-SPIR', mode = 'train'):
        self.mode = mode
        self.target_modality = target_modality
        self.json_paths = {
            'train': [
                './conditional_dataset/CHAOS_MRI_T1_InPhase_train.json',
                './conditional_dataset/CHAOS_MRI_T1_OutPhase_train.json',
                './conditional_dataset/CHAOS_MRI_T2_SPIR_train.json'
            ],
            'test': [
                './conditional_dataset/CHAOS_MRI_T1_InPhase_test.json',
                './conditional_dataset/CHAOS_MRI_T1_OutPhase_test.json',
                './conditional_dataset/CHAOS_MRI_T2_SPIR_test.json'
            ],
            'inference_T2-SPIR': [
                './conditional_dataset/CHAOS_MRI_T1_InPhase_all.json',
                './conditional_dataset/CHAOS_MRI_T1_OutPhase_all.json'
            ],
            'inference_T1': [
                './conditional_dataset/CHAOS_MRI_T2_SPIR_all.json'
            ]
        }

        self.data_list = []
        self.json_list = self._get_json_list()
        self._load_data()
    
    def _get_json_list(self):
        if self.mode == 'train':
            return self.json_paths['train']
        elif self.mode == 'test':
            return self.json_paths['test']
        elif self.mode == 'inference' and self.target_modality == 'T2-SPIR':
            return self.json_paths['inference_T2-SPIR']
        elif self.mode == 'inference' and self.target_modality in ['T1-InPhase', 'T1-OutofPhase']:
            return self.json_paths['inference_T1']
        return []

    def _load_data(self):
        for json_file in self.json_list:
            with open(json_file, 'r') as f:
                data = json.load(f)
            self.data_list.extend([item for item in data if len(item['organs']) > 0]) # filter samples without masks

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        image_path = data['image_path']
        mask_paths = [data.get(f'mask{i}_path', '') for i in range(4)]    # 4 organs for CHAOS, including liver, right kidney, left kidney, spleen
        image, masks = _process_image(data['image_path']), _process_mask(mask_paths)
        mask = torch.cat(masks, dim=0)
        merged_mask = sum([(m + 1) / 2. * f for m, f in zip(masks, [0.3, 0.5, 0.4, 0.6])]) * 2. - 1    # different weights for different organs
        
        modality, modality_attributes, region, label, organs = data['modality'], data['modality_attributes'], data['region'], ', '.join(data['label']), ', '.join(data['organs'])

        # ignore T2 SPIR
        if self.mode == 'train' and modality == 'T2 SPIR Abdomen MRI' and self.target_modality == 'T2-SPIR':
            mask = torch.zeros_like(mask) * 2 - 1.
            merged_mask = merged_mask * 0 - 1.
            organs = ''
        
        # ignore T1-InPhase and T1-OutofPhase
        if self.mode == 'train' and modality in ['T1 in-phase Abdomen MRI', 'T1 out-of-phase Abdomen MRI'] and self.target_modality in ['T1-InPhase', 'T1-OutofPhase']:
            mask = torch.zeros_like(mask) * 2 - 1.
            merged_mask = merged_mask * 0 - 1.
            organs = ''
         
        # prompt = ", ".join(filter(None, [modality, modality_attributes, region, organs])).strip()
        prompt = ", ".join(filter(None, [modality, modality_attributes, region, label])).strip()
        
        # Random Dropout Text Conditions
        if self.mode == 'train' and random.uniform(0, 1) < 0.1:
            prompt = ""
        
        return {"image_path": image_path, "image": image, "mask": mask, "merged_mask": merged_mask, "prompt": prompt, "organs": organs, "modality": modality, "region": region, "label": label, "modality_attributes": modality_attributes}


class Prostate_MRI_Dataset(Dataset):
    def __init__(self, target_modality = 'ADC', mode = 'train_MSD-MSD'):
        self.mode = mode
        self.target_modality = target_modality
        self.json_paths = {
            'train_MSD-MSD': [
                './conditional_dataset/MSD_Prostate_ADC_train.json',
                './conditional_dataset/MSD_Prostate_T2_train.json'
            ],
            'train_MSD-PROMISE': [
                './conditional_dataset/MSD_Prostate_ADC_train.json',
                './conditional_dataset/PROMISE12_T2_train.json'
            ],
            'test_MSD-MSD': [
                './conditional_dataset/MSD_Prostate_ADC_test.json',
                './conditional_dataset/MSD_Prostate_T2_test.json'
            ],
            'test_MSD-PROMISE': [
                './conditional_dataset/MSD_Prostate_ADC_test.json',
                './conditional_dataset/PROMISE12_T2_test.json'
            ],
            'inference_ADC-PROMISE': [
                './conditional_dataset/PROMISE12_T2_all.json'
            ],
            'inference_ADC-MSD': [
                './conditional_dataset/MSD_Prostate_T2_all.json'
            ],
            'inference_T2': [
                './conditional_dataset/MSD_Prostate_ADC_all.json'
            ]
        }

        self.data_list = []
        self.json_list = self._get_json_list()
        self._load_data()

    def _get_json_list(self):
        if self.mode in self.json_paths:
            return self.json_paths[self.mode]
        return []

    def _load_data(self):
        for json_file in self.json_list:
            with open(json_file, 'r') as f:
                data = json.load(f)
            self.data_list.extend([item for item in data if len(item['organs']) > 0])  # filter samples without masks

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        image_path = data['image_path']
        mask_paths = [data.get(f'mask{i}_path', '') for i in range(1)]
        image, masks = _process_image(data['image_path']), _process_mask(mask_paths)
        mask = torch.cat(masks, dim=0)
        merged_mask = sum([(m + 1) / 2. * f for m, f in zip(masks, [1.0])]) * 2. - 1
        modality, modality_attributes, region = data['modality'], data['modality_attributes'], data['region']
        label = data['label'][-1:] # for MSD-Prostate, it has 3 labels, but we only need the last one.
        label, organs = ', '.join(label), ', '.join(data['organs'])

        # generate ADC, so ignore ADC during training
        if self.mode == 'train' and modality == 'ADC Abdomen MRI' and self.target_modality == 'ADC':
            mask = torch.zeros_like(mask) * 2 - 1.
            merged_mask = merged_mask * 0 - 1.
            organs = ''
        
        # generate T2, so ignore T2 during training
        if self.mode == 'train' and modality == 'T2 Abdomen MRI' and self.target_modality == 'T2':
            mask = torch.zeros_like(mask) * 2 - 1.
            merged_mask = merged_mask * 0 - 1.
            organs = ''
         
        # prompt = ", ".join(filter(None, [modality, modality_attributes, region, organs])).strip()
        prompt = ", ".join(filter(None, [modality, modality_attributes, region, label])).strip()
        
        if self.mode == 'train' and random.uniform(0, 1) < 0.1:
            prompt = ""
        
        return {"image_path": image_path, "image": image, "mask": mask, "merged_mask": merged_mask, "prompt": prompt, "organs": organs, "modality": modality, "region": region, "label": label, "modality_attributes": modality_attributes}


class PanSeg_MRI_Dataset(Dataset): 
    def __init__(self, target_modality = 'T2', mode = 'train'):
        self.mode = mode
        self.target_modality = target_modality
        self.json_paths = {
            'train': [
                './conditional_dataset/PanSeg_T1_train.json',
                './conditional_dataset/PanSeg_T2_train.json'
            ],
            'test': [
                './conditional_dataset/PanSeg_T1_test.json',
                './conditional_dataset/PanSeg_T2_test.json'
            ],
            'inference_T2': [
                './conditional_dataset/PanSeg_T1_all.json'
            ],
            'inference_T1': [
                './conditional_dataset/PanSeg_T2_all.json'
            ]
        }

        self.data_list = []
        self.json_list = self._get_json_list()
        self._load_data()

    def _get_json_list(self):
        if self.mode == 'train':
            return self.json_paths['train']
        elif self.mode == 'test':
            return self.json_paths['test']
        elif self.mode == 'inference' and self.target_modality == 'T2':
            return self.json_paths['inference_T2']
        elif self.mode == 'inference' and self.target_modality == 'T1':
            return self.json_paths['inference_T1']
        return []

    def _load_data(self):
        for json_file in self.json_list:
            with open(json_file, 'r') as f:
                data = json.load(f)
            self.data_list.extend([item for item in data if len(item['organs']) > 0]) # filter samples without masks

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        image_path = data['image_path']
        mask_paths = [data.get(f'mask{i}_path', '') for i in range(1)] # only 1 organ for PanSeg
        image, masks = _process_image(data['image_path']), _process_mask(mask_paths)
        mask = torch.cat(masks, dim=0)
        merged_mask = sum([(m + 1) / 2. * f for m, f in zip(masks, [1.0])]) * 2. - 1
        modality, modality_attributes, region, label, organs = data['modality'], data['modality_attributes'], data['region'], ', '.join(data['label']), ', '.join(data['organs'])

        # synthesize T2, so ignore T2 during training
        if self.mode == 'train' and modality == 'T2 Abdomen MRI' and self.target_modality == 'T2':
            mask = torch.zeros_like(mask) * 2 - 1.
            merged_mask = merged_mask * 0 - 1.
            organs = ''
        
        # synthesize T1, so ignore T1 during training
        if  self.mode == 'train' and modality == 'T1 Abdomen MRI' and self.target_modality == 'T1':
            mask = torch.zeros_like(mask) * 2 - 1.
            merged_mask = merged_mask * 0 - 1.
            organs = ''
         
        # prompt = ", ".join(filter(None, [modality, modality_attributes, region, organs])).strip()
        prompt = ", ".join(filter(None, [modality, modality_attributes, region, label])).strip()
        
        if self.mode == 'train' and random.uniform(0, 1) < 0.1:
            prompt = ""
        
        return {"image_path": image_path, "image": image, "mask": mask, "merged_mask": merged_mask, "prompt": prompt, "organs": organs, "modality": modality, "region": region, "label": label, "modality_attributes": modality_attributes}
        

class LiQA_CHAOS_MRI_Dataset(Dataset): 
    def __init__(self, target_modality = 'T2-SPIR', mode = 'train'):
        self.mode = mode
        self.target_modality = target_modality
        self.json_paths = {
            'train': [
                './conditional_dataset/LiQA_GED4_train.json',
                './conditional_dataset/CHAOS_MRI_T2_SPIR_train.json'
            ],
            'test': [
                './conditional_dataset/LiQA_GED4_test.json',
                './conditional_dataset/CHAOS_MRI_T2_SPIR_test.json'
            ],
            'inference_T2-SPIR': [
                './conditional_dataset/LiQA_GED4_all.json'
            ],
            'inference_T1': [
                './conditional_dataset/CHAOS_MRI_T2_SPIR_all.json'
            ]
        }

        self.data_list = []
        self.json_list = self._get_json_list()
        self._load_data()

    def _get_json_list(self):
        if self.mode == 'train':
            return self.json_paths['train']
        elif self.mode == 'test':
            return self.json_paths['test']
        elif self.mode == 'inference' and self.target_modality == 'T2-SPIR':
            return self.json_paths['inference_T2-SPIR']
        elif self.mode == 'inference' and self.target_modality == 'T1':
            return self.json_paths['inference_T1']
        return []

    def _load_data(self):
        for json_file in self.json_list:
            with open(json_file, 'r') as f:
                data = json.load(f)
            # self.data_list.extend([item for item in data if len(item['organs']) > 0]) # filter samples without masks
            self.data_list.extend([item for item in data if 'liver' in item['organs']]) # filter samples without liver

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        image_path = data['image_path']
        mask_paths = [data.get(f'mask{i}_path', '') for i in range(1)]
        masks, image = _process_mask(mask_paths), _process_image(data['image_path'])
        mask = torch.cat(masks, dim=0)
        merged_mask = sum([(m + 1) / 2. * f for m, f in zip(masks, [1.0])]) * 2. - 1
        
        modality, modality_attributes, region, label, organs = data['modality'], data['modality_attributes'], data['region'], ', '.join(data['label']), 'liver'
        
        # ignore T2 SPIR
        if self.mode == 'train' and modality == 'T2 SPIR Abdomen MRI' and self.target_modality == 'T2-SPIR':
            mask = torch.zeros_like(mask) * 2 - 1.
            merged_mask = merged_mask * 0 - 1.
            organs = ''
        
        # ignore T1
        if self.mode == 'train' and modality == 'T1 Abdomen MRI' and self.target_modality == 'T1':
            mask = torch.zeros_like(mask) * 2 - 1.
            merged_mask = merged_mask * 0 - 1.
            organs = ''
        
        prompt = ", ".join(filter(None, [modality, modality_attributes, region, organs])).strip() # We only consider Liver here.
        
        if self.mode == 'train' and random.uniform(0, 1) < 0.1:
            prompt = ""
        
        return {"image_path": image_path, "image": image, "mask": mask, "merged_mask": merged_mask, "prompt": prompt, "organs": organs, "modality": modality, "region": region, "label": label, "modality_attributes": modality_attributes}

class MSD_Liver_CHAOS_MRI_Dataset(Dataset): 
    def __init__(self, target_modality = 'T2-SPIR', mode = 'train'):
        self.mode = mode
        self.target_modality = target_modality
        self.json_paths = {
            'train': [
                './conditional_dataset/MSD_Liver_train.json',
                './conditional_dataset/CHAOS_MRI_T2_SPIR_train.json'
            ],
            'test': [
                './conditional_dataset/MSD_Liver_test.json',
                './conditional_dataset/CHAOS_MRI_T2_SPIR_test.json'
            ],
            'inference_T2-SPIR': [
                './conditional_dataset/MSD_Liver_test.json'
            ],
            'inference_CT': [
                './conditional_dataset/CHAOS_MRI_T2_SPIR_all.json'
            ]
        }
        
        self.data_list = []
        self.json_list = self._get_json_list()
        self._load_data()
    
    def _get_json_list(self):
        if self.mode == 'train':
            return self.json_paths['train']
        elif self.mode == 'test':
            return self.json_paths['test']
        elif self.mode == 'inference' and self.target_modality == 'T2-SPIR':
            return self.json_paths['inference_T2-SPIR']
        elif self.mode == 'inference' and self.target_modality == 'CT':
            return self.json_paths['inference_CT']
        return []

    def _load_data(self):
        for json_file in self.json_list:
            with open(json_file, 'r') as f:
                data = json.load(f)
            # self.data_list.extend([item for item in data if len(item['organs']) > 0]) # filter samples without masks
            self.data_list.extend([item for item in data if 'liver' in item['organs']]) # filter samples without liver
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        image_path = data['image_path']
        mask_paths = [data.get(f'mask{i}_path', '') for i in range(1)]
        image, masks = _process_image(data['image_path']), _process_mask(mask_paths)
        mask = torch.cat(masks, dim=0)
        merged_mask = sum([(m + 1) / 2. * f for m, f in zip(masks, [1.0])]) * 2. - 1
        modality, modality_attributes, region, label, organs = data['modality'], data['modality_attributes'], data['region'], 'liver', 'liver'
        
        # ignore T2 SPIR
        if self.mode == 'train' and modality == 'T2-SPIR Abdomen MRI' and self.target_modality == 'T2-SPIR':
            mask = torch.zeros_like(mask) * 2 - 1.
            merged_mask = merged_mask * 0 - 1.
            organs = ''
        
        # ignore CT
        if self.mode == 'train' and modality == 'Abdomen CT' and self.target_modality == 'CT':
            mask = torch.zeros_like(mask) * 2 - 1.
            merged_mask = merged_mask * 0 - 1.
            organs = ''
        
        prompt = ", ".join(filter(None, [modality, modality_attributes, region, organs])).strip() # We only consider Liver here.

        if self.mode == 'train' and random.uniform(0, 1) < 0.1:
            prompt = ""
        
        return {"image_path": image_path, "image": image, "mask": mask, "merged_mask": merged_mask, "prompt": prompt, "organs": organs, "modality": modality, "region": region, "label": label, "modality_attributes": modality_attributes}


class AMOSCT_CHAOS_Dataset(Dataset):
    def __init__(self, target_modality = 'T2-SPIR', mode = 'train-T2'):
        self.mode = mode
        self.target_modality = target_modality
        self.json_paths = {
            'train-T2': [
                './conditional_dataset/AMOS22CT-CHAOS_train.json',
                './conditional_dataset/CHAOS_MRI_T2_SPIR_train.json'
            ],
            'train-T1': [
                './conditional_dataset/AMOS22CT-CHAOS_train.json',
                './conditional_dataset/CHAOS_MRI_T1_InPhase_train.json',
                './conditional_dataset/CHAOS_MRI_T1_OutPhase_train.json'
            ],
            'test-T2': [
                './conditional_dataset/AMOS22CT-CHAOS_test.json',
                './conditional_dataset/CHAOS_MRI_T2_SPIR_test.json'
            ],
            'test-T1': [
                './conditional_dataset/AMOS22CT-CHAOS_test.json',
                './conditional_dataset/CHAOS_MRI_T1_InPhase_test.json',
                './conditional_dataset/CHAOS_MRI_T1_OutPhase_test.json'
            ],
            'inference_T2-SPIR': [
                './conditional_dataset/AMOS22CT-CHAOS_test.json'
            ],
            'inference_CT-T2': [
                './conditional_dataset/CHAOS_MRI_T2_SPIR_all.json'
            ],
            'inference_CT-T1': [
                './conditional_dataset/CHAOS_MRI_T1_InPhase_all.json',
                './conditional_dataset/CHAOS_MRI_T1_OutPhase_all.json'
            ]
        }
        
        self.data_list = []
        self.json_list = self._get_json_list()
        self._load_data()
        
    def _get_json_list(self):
        if self.mode in self.json_paths:
            return self.json_paths[self.mode]
        return []

    def _load_data(self):
        for json_file in self.json_list:
            with open(json_file, 'r') as f:
                data = json.load(f)
            self.data_list.extend([item for item in data if len(item['organs']) > 0])  # filter samples without masks

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        image_path = data['image_path']
        mask_paths = [data.get(f'mask{i}_path', '') for i in range(4)] # 4 organs for CHAOS, including liver, right kidney, left kidney, spleen
        image, masks = _process_image(data['image_path']), _process_mask(mask_paths)
        mask = torch.cat(masks, dim=0)
        merged_mask = sum([(m + 1) / 2. * f for m, f in zip(masks, [0.3, 0.5, 0.4, 0.6])]) * 2. - 1
        modality, modality_attributes, region, label, organs = data['modality'], data['modality_attributes'], data['region'], ', '.join(data['label']), ', '.join(data['organs'])

        if self.mode == 'train-T2':
            # ignore T2 SPIR
            if self.mode == 'train-T2' and modality == 'T2 SPIR Abdomen MRI' and self.target_modality == 'T2-SPIR':
                mask = torch.zeros_like(mask) * 2 - 1.
                merged_mask = merged_mask * 0 - 1.
                organs = ''
            # ignore CT
            if self.mode == 'train-T2' and modality == 'Abdomen CT' and self.target_modality == 'CT':
                mask = torch.zeros_like(mask) * 2 - 1.
                merged_mask = merged_mask * 0 - 1.
                organs = ''
        
        elif self.mode == 'train-T1':
            # ignore T1
            if self.mode == 'train-T1' and modality in ['T1 in-phase Abdomen MRI', 'T1 out-of-phase Abdomen MRI'] and self.target_modality in ['T1-InPhase', 'T1-OutofPhase']:
                mask = torch.zeros_like(mask) * 2 - 1.
                merged_mask = merged_mask * 0 - 1.
                organs = ''
            # ignore CT
            if self.mode == 'train-T1' and modality == 'Abdomen CT' and self.target_modality == 'CT':
                mask = torch.zeros_like(mask) * 2 - 1.
                merged_mask = merged_mask * 0 - 1.
                organs = ''
            
        # prompt = ", ".join(filter(None, [modality, modality_attributes, region, organs])).strip()
        prompt = ", ".join(filter(None, [modality, modality_attributes, region, label])).strip()
        
        if self.mode == 'train' and random.uniform(0, 1) < 0.1:
            prompt = ""
        
        return {"image_path": image_path, "image": image, "mask": mask, "merged_mask": merged_mask, "prompt": prompt, "organs": organs, "modality": modality, "region": region, "label": label, "modality_attributes": modality_attributes}

if __name__ == '__main__':
    # train_dataset = CHAOS_MRI_Dataset(target_modality='T2-SPIR', mode='inference')
    # train_dataset = Prostate_MRI_Dataset(target_modality='ADC', mode='inference_T2')
    # train_dataset = PanSeg_MRI_Dataset(target_modality='T1', mode='train')
    # train_dataset = LiQA_CHAOS_MRI_Dataset(target_modality='T2-SPIR', mode='inference')
    # train_dataset = MSD_Liver_CHAOS_MRI_Dataset(target_modality='T2-SPIR', mode='inference')
    train_dataset = AMOSCT_CHAOS_Dataset(target_modality='T2-SPIR', mode='inference_T2-SPIR')
    print(train_dataset.__len__())
    train_data = DataLoader(train_dataset, batch_size=1, num_workers=64, shuffle=False)
    
    # B C H W
    for i, data in enumerate(train_data):
        print(i)
        print(data['image'].shape)
        print(data['mask'].shape)
        print(data['merged_mask'].shape)
