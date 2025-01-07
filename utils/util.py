import os
import sys
import cv2
import copy
import torch
import inspect
import datetime
import numpy as np
from PIL import Image
from os.path import *
from typing import Dict
from torchvision import transforms
from skimage.transform import resize, rotate

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_time_string() -> str:
    x = datetime.datetime.now()
    return f"{(x.year - 2000):02d}{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}{x.second:02d}"

def get_function_args() -> Dict:
    frame = sys._getframe(1)
    args, _, _, values = inspect.getargvalues(frame)
    args_dict = copy.deepcopy({arg: values[arg] for arg in args})

    return args_dict

# DataLoader utils
def normalize(image):
    image = np.array(image)
    min_val, max_val = np.min(image), np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val + 1e-6)
    normalized_image = Image.fromarray(normalized_image)
    
    return normalized_image
    
def _pad_to_square(image):
    image = np.array(image)
    height, width = image.shape[:2]
    
    if height > width:
        padding = (height - width) // 2
        padding_values = ((0, 0), (padding, height - width - padding))
    else:
        padding = (width - height) // 2
        padding_values = ((padding, width - height - padding), (0, 0))
    
    if len(image.shape) == 2:
        image_padded = np.pad(image, padding_values, mode='constant', constant_values=0)
    else:
        image_padded = np.pad(image, (padding_values, (0, 0)), mode='constant', constant_values=0)
    
    image_padded = Image.fromarray(image_padded)
    
    return image_padded

def _resize(image, target_size=(512, 512)):
    image = np.array(image)
    image = resize(image, target_size, preserve_range=True, anti_aliasing=True, mode='reflect')
    image = Image.fromarray(image)
    return image

def _rotate(image, angle=90):
    return rotate(image, angle, resize=True)

def _flip_horizontal(image):
    return np.fliplr(image).copy()

def _process_image(image_path):
    image = Image.open(image_path).convert('L')
    image = normalize(image) # [0, 1]
    image = _pad_to_square(image)
    image = _resize(image, target_size=(512, 512))
    
    return transforms.ToTensor()(image) * 2.0 - 1.0

def _process_mask(mask_paths):
    masks = []
    for path in mask_paths:
        mask = _process_image(path)
        if mask is not None:
            masks.append(mask)
        else:
            masks.append(torch.zeros(1, 512, 512)) - 1.
    return masks

# SAM2 utils
def calculate_iou(mask1, mask2):
    mask1, mask2 = np.where(mask1 > 0, 1, 0), np.where(mask2 > 0, 1, 0)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def get_bbox_from_mask(mask, add_noise=True):
    non_zero_indices = np.nonzero(mask)
    if non_zero_indices[0].size == 0:
        return None
    min_x, max_x = non_zero_indices[1].min(), non_zero_indices[1].max()
    min_y, max_y = non_zero_indices[0].min(), non_zero_indices[0].max()
    
    if add_noise:
        mask_y, mask_x = mask.shape
        noise_x_range = mask_x * 0.08
        noise_y_range = mask_y * 0.08
        
        min_x = min_x + np.random.uniform(-noise_x_range, noise_x_range)
        max_x = max_x + np.random.uniform(-noise_x_range, noise_x_range)
        min_y = min_y + np.random.uniform(-noise_y_range, noise_y_range)
        max_y = max_y + np.random.uniform(-noise_y_range, noise_y_range)
        
        min_x = np.clip(min_x, 0, mask_x)
        max_x = np.clip(max_x, 0, mask_x)
        min_y = np.clip(min_y, 0, mask_y)
        max_y = np.clip(max_y, 0, mask_y)
    
    bbox_mask = np.zeros_like(mask, dtype=np.float32)
    bbox_mask[int(min_y):int(max_y), int(min_x):int(max_x)] = 1.0
    
    return np.array([[min_x, min_y, max_x, max_y]]), bbox_mask

def get_center_point_from_mask(mask):
    non_zero_indices = np.nonzero(mask)
    if non_zero_indices[0].size == 0:
        return None
    mid_x = (non_zero_indices[1].min() + non_zero_indices[1].max()) // 2
    mid_y = (non_zero_indices[0].min() + non_zero_indices[0].max()) // 2
    return np.array([[mid_x, mid_y]])

def save_images(image_array, filenames, directory):
    """Save images from an array to specified filenames and directory."""
    output_image_array = []
    for img, fname in zip(image_array, filenames):
        img_path = os.path.join(directory, fname)
        img = (img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1.) / 2.
        cv2.imwrite(img_path, img[:, :, ::-1] * 255)
        output_image_array.append(img.transpose(2, 0, 1))
    
    return output_image_array