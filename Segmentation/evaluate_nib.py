import numpy as np
import cv2
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
import json
import nibabel as nib
join = os.path.join

from record_results import record

dataset2label_mapping = {
    'CHAOSMR' : {
            "liver": 1,
            "right kidney": 2,
            "left kidney": 3,
            "spleen": 4,
            "kidney": [2, 3]
    },
    'AMOS22_CT' : {
            "liver": 6,
            "right kidney": 2,
            "left kidney": 3,
            "spleen": 1,
            "kidney": [2, 3]
    },
    'MSD_Prostate' : {
        "transition zone of prostate": 1,
        "peripheral zone of prostate": 2,
        "prostate": [1, 2],
    },
    'Prostate' : {
        "prostate": 1,
    },
    'Liver' : {
        "liver": 1,
    },
    'Pancreas' : {
        "pancreas": 1,
    },
    'Liver Tumor' : {
        "liver": [1,2],
        "liver tumor":2,
    },
}

nnUNet_RAW = 'Your Path to nnUNet Raw Data'

def calculate_metrics(gt_dir, seg_dir, img_dir, nnunet_name, target_dataset, source_dataset):

    print(f'Eval from {source_dataset} to {target_dataset}')
    print(f'Pred {seg_dir}, GT {gt_dir}, {nnunet_name}')
    
    img_path = join('{}/{}/{}'.format(nnUNet_RAW, nnunet_name, img_dir))
    gt_path = join('{}/{}/{}'.format(nnUNet_RAW, nnunet_name, gt_dir))
    seg_path = '{}/{}/{}'.format(nnUNet_RAW, nnunet_name, seg_dir)
    csv_path = '{}/{}/{}/(NII){}_{}.csv'.format(nnUNet_RAW, nnunet_name, seg_dir, nnunet_name, seg_dir)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(join('{}/{}/dataset.json'.format(nnunet_name)), 'r') as f:
        tmp = json.load(f)
        modality = tmp["channel_names"]['0']
        
    modality = {
        'CT':'ct',
        'MR':'mri',
        'PET':'pet'
    }[modality]

    filenames = np.sort(os.listdir(seg_path))
    filenames = [x for x in filenames if x.endswith('.png')]

    labels_in_gt = dataset2label_mapping[target_dataset]
    labels_in_pred = dataset2label_mapping[source_dataset]
    
    # find all volumes and their slices
    data = {}
    for name in filenames:
        # combine 2D grond truth and segmentation into 3D
        # name = {case_id}_s{slice_index}.png
        
        tmp = name.split('_s')
        case_id = '_s'.join(tmp[:-1])
        slice_index = tmp[-1].replace('.png', '')
        if case_id not in data.keys():
            data[case_id] = []
        
        data[case_id].append(int(slice_index))
        
    results_of_samples = []
    
    for case_id, datum in data.items():

        scores = []
        labels = []
        gt_data = []
        seg_data = []
        img_data = []
        
        data[case_id] = sorted(data[case_id])
        for slice_idx in data[case_id]:
            
            tmp_gt = cv2.imread(join(gt_path, '{}_s{}.png'.format(case_id, slice_idx)), -1)
            assert tmp_gt.shape == (512, 512), f"{join(gt_path, '{}_s{}.png'.format(case_id, slice_idx))} : {tmp_gt.shape}"
            gt_data.append(tmp_gt)
            
            tmp_seg = cv2.imread(join(seg_path, '{}_s{}.png'.format(case_id, slice_idx)), -1)
            assert tmp_seg.shape == (512, 512), f"{join(seg_path, '{}_s{}.png'.format(case_id, slice_idx))} : {tmp_seg.shape}"
            seg_data.append(tmp_seg)
            
            img_data.append(cv2.imread(join(img_path, '{}_s{}_0000.png'.format(case_id, slice_idx)), -1))
        
        gt_data = np.stack(gt_data, axis=-1) # 512 512 D
        seg_data = np.stack(seg_data, axis=-1) # 512 512 D
        img_data = np.stack(img_data, axis=-1) # 512 512 D

        for organ in labels_in_gt.keys():
            if organ not in labels_in_pred:
                continue
            
            ind_in_gt = labels_in_gt[organ]
            ind_in_pred = labels_in_pred[organ]
            
            if isinstance(ind_in_gt, int):
                organ_gt = gt_data==ind_in_gt
            else:
                organ_gt = np.zeros(gt_data.shape, dtype=np.uint8)
                for i in ind_in_gt:
                    organ_gt += gt_data==i
                organ_gt = organ_gt > 0
                
            if isinstance(ind_in_pred, int):
                organ_seg = seg_data==ind_in_pred
            else:
                organ_seg = np.zeros(seg_data.shape, dtype=np.uint8)
                for i in ind_in_pred:
                    organ_seg += seg_data==i
                organ_seg = organ_seg > 0

            if np.sum(organ_gt)==0 and np.sum(organ_seg)==0:
                DSC = 1
                NSD = 1
            elif np.sum(organ_gt)==0 and np.sum(organ_seg)>0:
                DSC = 0
                NSD = 0
            else:
                DSC = compute_dice_coefficient(organ_gt, organ_seg)
                # surface_distances = compute_surface_distances(organ_gt, organ_seg, case_spacing)
                # NSD = compute_surface_dice_at_tolerance(surface_distances, 1)
            
            scores.append({'dice':round(DSC, 4), 'nsd':0})
            labels.append(organ)
            
        gt_obj = nib.nifti2.Nifti1Image(gt_data, np.eye(4))
        nib.save(gt_obj, '{}/{}/{}/{}.nii.gz'.format(nnUNet_RAW, nnunet_name, gt_dir, case_id))
        
        seg_obj = nib.nifti2.Nifti1Image(seg_data, np.eye(4))
        nib.save(seg_obj, '{}/{}/{}/{}.nii.gz'.format(nnUNet_RAW, nnunet_name, seg_dir, case_id))
        
        img_obj = nib.nifti2.Nifti1Image(img_data, np.eye(4))
        nib.save(img_obj, '{}/{}/{}/{}_0000.nii.gz'.format(nnUNet_RAW, nnunet_name, img_dir, case_id))
            
        results_of_samples.append([target_dataset, modality, case_id, scores, labels])
        
    record(results_of_samples, csv_path)
    
    print(f'Save results to {csv_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute Dice scores')
    parser.add_argument('--target_dataset', type=str, default='CHAOS_MRI', help='Name of the target dataset')
    parser.add_argument('--source_dataset', type=str, default='CHAOS_MRI', help='Name of the source dataset')
    parser.add_argument('--nnunet_name', type=str, default='Dataset995_CHAOSMR_T2SPIR', help='nnUNet name of the source dataset')
    parser.add_argument('--img_dir', type=str, default='imagesTs', help='dir name of test images')
    parser.add_argument('--gt_dir', type=str, default='labelsTs', help='dir name of gt segmentations')
    parser.add_argument('--seg_dir', type=str, default='labelsPred', help='dir name of prediction segmentations')
    
    args = parser.parse_args()
    
    calculate_metrics(args.gt_dir, args.seg_dir, args.img_dir, args.nnunet_name, args.target_dataset, args.source_dataset)
    