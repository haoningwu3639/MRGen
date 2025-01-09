import os
import cv2
import json
import torch
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import AutoTokenizer, AutoModel
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available

from stage2_dataset import AMOSCT_CHAOS_Dataset
from utils.util import calculate_iou, save_images
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from model.controlnet import ControlNetModel
from model.autoencoder_kl import AutoencoderKL
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline_controlnet import StableDiffusionControlNetPipeline

logger = get_logger(__name__)

def setup_pipeline(pretrained_model_path):
    """Setup the pipeline with VAE, text encoder, UNet, and ControlNet."""
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = AutoModel.from_pretrained(os.path.join(pretrained_model_path, "text_encoder"), trust_remote_code=True)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    controlnet = ControlNetModel.from_pretrained(pretrained_model_path, subfolder="controlnet")

    pipeline = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)
    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            logger.warning("Could not enable memory efficient attention. Make sure xformers is installed")

    print("Models loaded successfully.")
    return pipeline

def save_json(target_list, target_json_path):
    """Save the target list to a JSON file."""
    sorted_list = sorted(target_list, key=lambda x: x['index'])
    with open(target_json_path, 'w') as target_json:
        json.dump(sorted_list, target_json, indent=4, ensure_ascii=False)

def filter_json(input_json_path, output_json_path, iou_threshold_input, confidence_threshold_input, avg_iou_threshold_input, avg_confidence_threshold_input):
    """Filter the JSON file to keep only the top 2 samples with the highest average IoU for each index."""
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    filtered_data = []
    index_dict = {}

    for item in data:
        index = item['index']
        if index not in index_dict:
            index_dict[index] = []
        index_dict[index].append(item)
    
    for index, items in index_dict.items():
        iou_threshold, confidence_threshold, avg_iou_threshold, avg_confidence_threshold = iou_threshold_input, confidence_threshold_input, avg_iou_threshold_input, avg_confidence_threshold_input
        
        items.sort(key=lambda x: x['sam2_avg_iou'], reverse=True)
        top_items = items[:5] # We select the top 5 samples with the highest average IoU
        
        def check_conditions(item, iou_threshold, confidence_threshold, avg_iou_threshold, avg_confidence_threshold):
            if item['sam2_avg_confidence'] < avg_confidence_threshold:
                return False
            if item['sam2_avg_iou'] < avg_iou_threshold:
                return False
            if any(iou < iou_threshold for iou in item['sam2_organs_iou'] if iou != 0):
                return False
            if any(confidence < confidence_threshold for confidence in item['sam2_organs_confidence'] if confidence != 0):
                return False
            return True
        
        candidates = [item for item in top_items if check_conditions(item, iou_threshold, confidence_threshold, avg_iou_threshold, avg_confidence_threshold)]
        
        # If we have less than 2 candidates, we reduce the thresholds by 0.10 and try again
        if len(candidates) < 2:
            iou_threshold -= 0.10
            confidence_threshold -= 0.10
            avg_iou_threshold -= 0.10
            avg_confidence_threshold -= 0.10
            candidates = [item for item in top_items if check_conditions(item, iou_threshold, confidence_threshold, avg_iou_threshold, avg_confidence_threshold)]
            filtered_data.extend(candidates[:1])
        
        else:
            filtered_data.extend(candidates[:2])
    
    with open(output_json_path, 'w') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

def test(rank, world_size, args, shared_list):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    checkpoint, model_cfg = "./checkpoints/sam2_hiera_large.pt", "sam2_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    
    args_dict = vars(args)
    if rank == 0:
        print(args_dict)
        json_path = os.path.join(args.log_dir, "args_config.json")
        with open(json_path, 'w') as json_file:
            json.dump(args_dict, json_file, indent=4)
    
    pretrained_model_path = args.ckpt
    
    logdirs = {
        'json_dir': os.path.join(args.log_dir, 'json_files'),
        'mask': os.path.join(args.log_dir, 'synthetic_mask'),
        'image': os.path.join(args.log_dir, f'synthetic_{args.modality}'),
        'combined': os.path.join(args.log_dir, f'synthetic_{args.modality}_combined')
    }
    
    target_json_path = os.path.join(args.log_dir, 'json_files', f'synthetic_{args.modality}.json')
    
    if rank == 0:
        for dir_path in logdirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    mixed_precision = "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    pipeline = setup_pipeline(pretrained_model_path)
    vae, text_encoder, unet, controlnet = pipeline.vae, pipeline.text_encoder, pipeline.unet, pipeline.controlnet
    for model in [vae, text_encoder, unet, controlnet]:
        model.requires_grad_(False)
        model.eval()
    
    val_dataset = AMOSCT_CHAOS_Dataset(modality=args.modality, mode='inference')
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, sampler=val_sampler)
    
    if rank == 0:
        print(f"Dataset length: {len(val_dataset)}")
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)
    
    progress_bar = tqdm(val_dataloader, desc="Validating", leave=False, disable=rank != 0)
    indices = list(range(len(val_dataset)))
    indices = indices[rank::world_size]  # split the indices for each sub-process
    
    for i, data in enumerate(progress_bar):
        if i >= len(indices):
            break
        # obtain the absolute index of the data
        absolute_index = indices[i]
        
        print(f"Rank {rank} is processing data index {absolute_index}")
        # check if the sample already exists in the shared list
        existing_samples = [item for item in shared_list if item['index'] == absolute_index]
        if len(existing_samples) >= args.sample_per_mask:
            continue
        
        image = data["image"].to(dtype=weight_dtype, device=device)
        mask = data["mask"].to(dtype=weight_dtype, device=device)
        merged_mask = data["merged_mask"].to(dtype=weight_dtype, device=device)
        
        if args.modality == 'CT-T1' or args.modality == 'CT-T2':
            prompt = [", ".join(['Abdomen CT, fat low signal, muscle moderate signal, water low signal, fat dark, muscle gray, water dark', data['region'][0], data['label'][0]])]
        elif args.modality == 'T2-SPIR':
            prompt = [", ".join(['T2 SPIR Abdomen MRI, fat low signal, muscle low signal, water high signal, fat dark, water bright', data['region'][0], data['label'][0]])]  
        elif args.modality == 'T1-InPhase':
            prompt = [", ".join(['T1 in-phase Abdomen MRI, fat high signal, muscle intermediate signal, water low signal, fat bright', data['region'][0], data['label'][0]])]
        elif args.modality == 'T1-OutPhase':
            prompt = [", ".join(['T1 out-of-phase Abdomen MRI, fat low signal, muscle intermediate signal, water低 signal, fat dark', data['region'][0], data['label'][0]])]
        
        merged_mask_img = (merged_mask + 1.) / 2.
        merged_mask_img = merged_mask_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        
        if np.mean(np.where(merged_mask_img > 0, 1, 0)) <= args.mask_threshold:
            continue
        
        cv2.imwrite(os.path.join(logdirs['mask'], f"{absolute_index}_mask.png"), merged_mask_img[:, :, ::-1] * 255)
        merged_mask_img = Image.fromarray((merged_mask_img[:, :, 0] * 255).astype(np.uint8)).convert('L')
        mask_images = [mask[:, [j], :, :] for j in range(4)]
        mask_file_names = [f"{absolute_index}_mask_{j}.png" for j in range(4)]
        mask_images = save_images(mask_images, mask_file_names, logdirs['mask'])

        sample_seeds = torch.randint(0, 100000, (args.max_tries_per_mask,)).tolist()
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in sample_seeds]
        
        for j in range(0, len(generators), args.num_images_per_prompt):
            current_generators = generators[j:j + args.num_images_per_prompt]
            sequence = pipeline(
                prompt=prompt,
                image=mask if args.channels == 4 else merged_mask,
                height=image.shape[2],
                width=image.shape[3],
                generator=current_generators,
                num_inference_steps=50,
                guidance_scale=7.0,
                num_images_per_prompt=args.num_images_per_prompt,
            ).images

            for k, output_image in enumerate(sequence):

                image_array = (np.repeat(np.array(output_image)[None, :, :], 3, axis=0).transpose(1, 2, 0) * 255.).astype(np.uint8)
                organs_iou, organs_confidence = [], []

                for m in range(len(mask_images)):
                    initial_mask = mask_images[m]
                    initial_mask_image = Image.fromarray((initial_mask * 255.).astype(np.uint8).squeeze(), mode='L')
                    if np.mean(initial_mask) == 0:
                        organs_iou.append(0.0)
                        organs_confidence.append(0.0)
                        continue

                    resized_mask_image = initial_mask_image.resize([256, 256])
                    resized_mask = np.array(resized_mask_image) 

                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        predictor.set_image(image_array)
                        masks, scores, logits = predictor.predict(mask_input=resized_mask[None, :, :])
                        
                    best_mask_index = np.argmax(scores)
                    sam_mask, confidence = masks[best_mask_index], scores[best_mask_index]
                    iou = calculate_iou(initial_mask, sam_mask)

                    organs_iou.append(iou)
                    organs_confidence.append(confidence)

                non_zero_organs_iou = [iou for iou in organs_iou if iou != 0]
                non_zero_organs_confidence = [confidence for confidence in organs_confidence if confidence != 0]
                non_zero_iou = np.mean(non_zero_organs_iou)
                non_zero_confidence = np.mean(non_zero_organs_confidence)

                item = {
                    'index': absolute_index,
                    'prompt': prompt,
                    'region': data['region'][0],
                    'organs': data['organs'][0],
                    'label': data['label'][0],
                    'mask_path': os.path.join(logdirs['mask'], f"{absolute_index}_mask.png"),
                    'image_path': os.path.join(logdirs['image'], f"{absolute_index}_{args.modality}_Abdomen_MRI_{j+k}.png"),
                    'combined_path': os.path.join(logdirs['combined'], f"{absolute_index}_T1-{args.modality}_Abdomen_MRI_{j+k}_combined.png"),
                    'sam2_avg_iou': float(non_zero_iou),
                    'sam2_avg_confidence': float(non_zero_confidence),
                    'sam2_organs_iou': [float(item) for item in organs_iou],
                    'sam2_organs_confidence': [float(item) for item in organs_confidence]
                }
                item.update({f"mask_{m}_path": os.path.join(logdirs['mask'], f"{absolute_index}_mask_{m}.png") for m in range(4)})

                output_image.save(item['image_path'])
                combined_img = Image.new('L', (output_image.width + merged_mask_img.width, output_image.height))

                combined_img.paste(merged_mask_img, (0, 0))
                combined_img.paste(output_image, (merged_mask_img.width, 0))
                combined_img.save(item['combined_path'])

                shared_list.append(item)
                
                # if rank == 0:
                #     save_json(list(shared_list), target_json_path)
            
        if rank == 0:
            save_json(list(shared_list), target_json_path)
    
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--modality', type=str, choices=['CT-T1', 'CT-T2', 'T2-SPIR', 'T1-InPhase', 'T1-OutPhase'], required=True)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--sample_per_mask', type=int, default=2)
    parser.add_argument('--num_images_per_prompt', type=int, default=10)
    parser.add_argument('--max_tries_per_mask', type=int, default=20)
    parser.add_argument('--mask_threshold', type=float, default=0.005)
    parser.add_argument('--iou_threshold', type=float, default=0.70)
    parser.add_argument('--confidence_threshold', type=float, default=0.80)
    parser.add_argument('--avg_iou_threshold', type=float, default=0.80)
    parser.add_argument('--avg_confidence_threshold', type=float, default=0.90)
    parser.add_argument('--num_gpus', type=int, default=4, help='number of GPUs to use')
    
    args = parser.parse_args()
    world_size = args.num_gpus

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)

    # create a shared list to store the results
    manager = mp.Manager()
    shared_list = manager.list()

    mp.spawn(test, args=(world_size, args, shared_list), nprocs=world_size, join=True)

    # save the shared list to a JSON file
    save_json(list(shared_list), os.path.join(args.log_dir, 'json_files', f'synthetic_{args.modality}.json'))
    
    filter_json(
        input_json_path=os.path.join(args.log_dir, 'json_files', f'synthetic_{args.modality}.json'),
        output_json_path=os.path.join(args.log_dir, 'json_files', f'synthetic_{args.modality}_filtered.json'),
        iou_threshold_input=args.iou_threshold,
        confidence_threshold_input=args.confidence_threshold,
        avg_iou_threshold_input=args.avg_iou_threshold,
        avg_confidence_threshold_input=args.avg_confidence_threshold
    )

if __name__ == "__main__":
    main()