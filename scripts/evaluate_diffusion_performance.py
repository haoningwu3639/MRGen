import os
import json
import torch
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import ConcatDataset, DataLoader
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, AutoModel
from model.autoencoder_kl import AutoencoderKL
from model.pipeline import StableDiffusionPipeline
from model.unet_2d_condition import UNet2DConditionModel
from stage1_dataset import MRGenDataset

logger = get_logger(__name__)

def load_test_data():
    test_CT_dataset = MRGenDataset(data_json_file='./radiopaedia_abdomen_ct_image_annotated.json', mode='test', stage='vae', modality='CT')
    test_MRI_dataset = MRGenDataset(data_json_file='./radiopaedia_abdomen_mri_image_annotated.json', mode='test', stage='vae', modality='MRI')
    
    val_dataset = ConcatDataset([test_CT_dataset, test_MRI_dataset])
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)
    
    return val_dataloader

def setup_pipeline(pretrained_model_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = AutoModel.from_pretrained(os.path.join(pretrained_model_path, "text_encoder"), trust_remote_code=True)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning("Could not enable memory efficient attention. Make sure xformers is installed")
    
    return pipeline

def save_json(target_list, target_json_path):
    sorted_list = sorted(target_list, key=lambda x: x['index'])
    with open(target_json_path, 'w') as target_json:
        json.dump(sorted_list, target_json, indent=4, ensure_ascii=False)

def test(rank, world_size, args, shared_list):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    pretrained_model_path, logdir = args.ckpt, args.log_dir
    mixed_precision = "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    pipeline = setup_pipeline(pretrained_model_path)
    vae, text_encoder, unet = pipeline.vae, pipeline.text_encoder, pipeline.unet
    for model in [vae, text_encoder, unet]:
        model.requires_grad_(False)
        model.eval()
    
    val_dataset = load_test_data().dataset
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, sampler=val_sampler)
    
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
    
    progress_bar = tqdm(val_dataloader, desc="Validating", leave=False, disable=rank != 0)
    
    indices = list(range(len(val_dataset)))
    indices = indices[rank::world_size]  # split the dataset to each GPU
    
    for i, data in enumerate(progress_bar):
        if i >= len(indices):
            break
        absolute_index = indices[i]
        print(f"Rank {rank} is processing data index {absolute_index}")

        original_path, prompt, modality, aux_modality, region, organs = data['image_path'][0], data['prompt'][0], data['modality'][0], data['aux_modality'][0], data['region'][0], data['organs'][0]
        
        # Generate new file name
        parts = original_path.split('/')
        image_name = '-'.join(parts[-4:]).replace('/', '-').rsplit('.', 1)[0] + '.png'
        image_path = os.path.join(logdir, image_name)
        
        random_seed = random.randint(0, 100000)
        generator = torch.Generator(device=device).manual_seed(random_seed)
        images = pipeline(
            prompt,
            height=512,
            width=512,
            generator=generator,
            num_inference_steps=50,
            guidance_scale=7.0,
            num_images_per_prompt=1,
        ).images[0]
        
        # Save image
        images[0].save(image_path)
        
        # Append result
        shared_list.append({
            'index': absolute_index,
            'original_path': original_path,
            'image_path': image_path,
            'prompt': prompt,
            'modality': modality,
            'aux_modality': aux_modality,
            'region': region,
            'organs': organs
        })
    
    if rank == 0:
        save_json(list(shared_list), os.path.join(logdir, './radiopaedia_synthetic_results_MRGen.json'))
    
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, default=4, help='number of GPUs to use')
    
    args = parser.parse_args()
    world_size = args.num_gpus
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)

    manager = mp.Manager()
    shared_list = manager.list()

    mp.spawn(test, args=(world_size, args, shared_list), nprocs=world_size, join=True)

    save_json(list(shared_list), os.path.join(args.log_dir, './radiopaedia_synthetic_results_MRGen.json'))

if __name__ == "__main__":
    main()