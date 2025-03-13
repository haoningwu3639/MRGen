import json
import torch
import random
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class MRGenDataset(Dataset):
    def __init__(self, data_json_file, mode='train', stage='vae', modality='CT'):
        self.mode = mode
        self.stage = stage
        self.data_json_file = data_json_file
        self.modality_attibutes_json = './modality_attributes.json'
        self.modality = modality
        
        if self.stage == 'vae':
            self.transform = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.RandomApply([
                    transforms.RandomRotation(90, expand=False),
                    transforms.RandomRotation(180, expand=False),
                    transforms.RandomRotation(270, expand=False)
                ], p=0.2) 
            ])
        
        with open(self.data_json_file, 'r') as f:
            self.data_list = json.load(f)
        
        with open(self.modality_attibutes_json, 'r') as f:
            self.modality_attibutes = json.load(f)
        
        train_mode_num = int(len(self.data_list) * 0.99)
        if self.mode == 'train':
            self.data_list = self.data_list[:train_mode_num]
        elif self.mode == 'test':
            self.data_list = self.data_list[train_mode_num:]
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        image_path, aux_modality, image_modality, region, organs = data['image_path'], data.get('aux_modality', ''), data.get('image_modality', ''), data['region'], data['organs']
        
        modality_key = " ".join([aux_modality, 'Abdomen', image_modality]).strip() if self.modality == 'MRI' else " ".join(['Abdomen', image_modality]).strip()
        modality_attributes = self.modality_attibutes[modality_key]
        
        prompt = [modality_key, modality_attributes, region, organs]
        prompt = ", ".join(filter(None, prompt))
        
        image = Image.open(image_path).convert('L')
        width, height = image.size
        # padding to square
        padding = (0, (width - height) // 2, 0, (width - height) - (width - height) // 2) if width > height else ((height - width) // 2, 0, (height - width) - (height - width) // 2, 0)
        image = ImageOps.expand(image, padding, fill=0)
        image = image.resize((512, 512))
        image = transforms.ToTensor()(image)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        image = image * 2. - 1.  # normalize
        
        if self.mode == 'train':
            # Data Augmentation
            if self.stage == 'vae':
                image = self.transform(image)
            # Random Dropout Text Condition
            if random.uniform(0, 1) < 0.1:
                prompt = ''

        return {"image_path": image_path, "image": image, 'modality': image_modality, 'aux_modality': aux_modality, 'prompt': prompt, 'region': region, 'organs': organs}


if __name__ == '__main__':
    train_MRI_dataset = MRGenDataset(data_json_file='./radiopaedia_abdomen_mri_image_annotated.json', mode='train', stage='vae', modality='MRI')
    test_MRI_dataset = MRGenDataset(data_json_file='./radiopaedia_abdomen_mri_image_annotated.json', mode='test', stage='vae', modality='MRI')
    
    train_dataset = train_MRI_dataset
    test_dataset = test_MRI_dataset
    
    train_data = DataLoader(train_dataset, batch_size=64, num_workers=64, shuffle=False)
    test_data = DataLoader(test_dataset, batch_size=64, num_workers=64, shuffle=False)
    
    for i, data in enumerate(train_data):
        print(i)
        # print(data['image_path'])
        # print(data['image'].shape)
        # print(data['prompt'])