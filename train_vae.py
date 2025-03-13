import os
import wandb
import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from torch.optim import AdamW
from omegaconf import OmegaConf
from stage1_dataset import MRGenDataset
from torchvision.utils import save_image
from diffusers.optimization import get_scheduler
from utils.util import get_function_args, get_time_string
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from model.autoencoder_kl import AutoencoderKL, vae_compute_loss

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9999'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
    torch.cuda.empty_cache()

@torch.no_grad()
def validate(vae, test_dataloader, device, kld_weight: float, logdir: str, step: int):
    print("Validating...")
    vae.eval()
    val_loss = 0.0
    total_batches = len(test_dataloader)
    progress_bar = tqdm(test_dataloader, desc="Validating", leave=False)
    images_to_save = []

    for batch_idx, batch in enumerate(progress_bar):
        image = batch["image"].to(device=device, dtype=torch.float32)
        
        if image.shape[1] == 1 and vae.module.in_channels == 3:
            image = image.repeat(1, 3, 1, 1)
        
        loss_dict = vae_compute_loss(kld_weight, vae(image, sample_posterior=True, return_dict=False))
        batch_loss = loss_dict['recon_l2_loss'].item()
        val_loss += batch_loss
        progress_bar.set_postfix(val_loss=val_loss / (batch_idx + 1))

        # Collect images for saving
        if batch_idx < 8:  # Adjust based on how many batches you want to save
            recon_images = vae(image, sample_posterior=True, return_dict=False)[0]
            images_to_save.append((image.cpu(), recon_images.cpu()))

    # Save images
    if images_to_save:
        original_images = torch.cat([img[0] for img in images_to_save], dim=0)[:8]
        reconstructed_images = torch.cat([img[1] for img in images_to_save], dim=0)[:8]
        # Normalize the images from [-1, 1] to [0, 1]
        original_images = (original_images + 1) / 2
        reconstructed_images = (reconstructed_images + 1) / 2
        # Save the images in a grid
        save_image(original_images, os.path.join(logdir, f'ground_truth_{step}.png'), nrow=4)
        save_image(reconstructed_images, os.path.join(logdir, f'reconstructed_{step}.png'), nrow=4)

    return val_loss / total_batches

def make_data_yielder(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def train(rank, world_size, config):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        wandb.init(project=config.name, mode="offline")
        logdir = config.logdir + f"_{get_time_string()}"
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(get_function_args(), os.path.join(logdir, 'config.yml'))

    train_MRI_dataset = MRGenDataset(data_json_file=config.data_mri_json_file, mode='train', stage='vae', modality='MRI')
    test_MRI_dataset = MRGenDataset(data_json_file=config.data_mri_json_file, mode='test', stage='vae', modality='MRI')
    
    train_dataset = train_MRI_dataset
    test_dataset = test_MRI_dataset

    print(train_dataset.__len__())
    print(test_dataset.__len__())
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, sampler=train_sampler, pin_memory=True, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=config.val_batch_size, pin_memory=True, num_workers=16, shuffle=False)

    if config.continue_train_path:
        vae = AutoencoderKL.from_pretrained(config.continue_train_path, subfolder='vae')
    else:
        vae = AutoencoderKL.from_config(config.pretrained_model_path, subfolder='vae')
    
    vae.to(device)
    vae = torch.nn.parallel.DistributedDataParallel(vae, device_ids=[rank])
    
    optimizer = AdamW(
        vae.module.parameters(), 
        lr=config.learning_rate, 
        betas= (config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon
    )
    
    lr_scheduler = get_scheduler(config.lr_scheduler, optimizer, num_warmup_steps= config.lr_warmup_steps, num_training_steps= config.train_steps)
    val_loss = 1.0
    train_data_yielder = make_data_yielder(train_dataloader)
    
    if rank == 0:
        progress_bar = tqdm(range(config.train_steps), desc="Steps")
        train_loss_record = {'loss': [], 'recon_l2_loss': [], 'kld_loss': []}
        val_loss_record = []
        
    for step in range(config.train_steps):
        vae.train()
        for accumulation_index in range(config.gradient_accumulation_steps):
            batch = next(train_data_yielder)
            image = batch["image"].to(device=device)
            if image.shape[1] == 1 and vae.module.in_channels == 3:
                image = image.repeat(1, 3, 1, 1)
            
            loss_dict = vae_compute_loss(config.kld_weight, vae(image, sample_posterior=True))
            loss = loss_dict['loss'] / config.gradient_accumulation_steps
            loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        
        if rank == 0:
            progress_bar.update(1)
            progress_bar.set_postfix(train_loss=loss_dict['loss'].item(), 
                                    val_loss=val_loss,
                                    kld_loss=loss_dict['kld_loss'].item(), 
                                    recon_l2_loss=loss_dict['recon_l2_loss'].item(), 
                                    lr=lr_scheduler.get_last_lr()[0])
            train_loss_record['loss'].append(loss_dict['loss'].item())
            train_loss_record['recon_l2_loss'].append(loss_dict['recon_l2_loss'].item())
            train_loss_record['kld_loss'].append(loss_dict['kld_loss'].item())

            wandb.log({
                    "Train Loss": loss_dict['loss'].item(),
                    "KL Divergence Loss": loss_dict['kld_loss'].item(),
                    "Reconstruction L2 Loss": loss_dict['recon_l2_loss'].item(),
                    "Learning Rate": lr_scheduler.get_last_lr()[0],
                }, step=step)
            
            if step % (config.checkpointing_steps // 10) == 0:
                checkpoint_save_path = os.path.join(logdir, f"checkpoint_lastest")
                os.makedirs(checkpoint_save_path, exist_ok=True)
                vae.module.save_pretrained(checkpoint_save_path)
            
            if step % config.checkpointing_steps == 0:
                checkpoint_save_path = os.path.join(logdir, f"checkpoint_{step}")
                os.makedirs(checkpoint_save_path, exist_ok=True)
                vae.module.save_pretrained(checkpoint_save_path)
                
                np.savez(
                    os.path.join(checkpoint_save_path, 'train_loss_record'),
                    loss=np.array(train_loss_record['loss']),
                    recon_l2_loss=np.array(train_loss_record['recon_l2_loss']),
                    kld_loss=np.array(train_loss_record['kld_loss'])
                )
                np.save(os.path.join(checkpoint_save_path, 'val_loss_record'), np.array(val_loss_record))
                
                wandb.log({"Validation Loss": val_loss}, step=step)
                
            if (step + 1) % config.validation_steps == 0:
                torch.cuda.empty_cache()
                val_loss = validate(vae, test_dataloader, device, config.kld_weight, logdir, step)
                torch.cuda.empty_cache()
                print(f"Validation loss: {val_loss}")
                val_loss_record.append(val_loss)
                vae.train()
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    config = OmegaConf.load("./training_configs/train_vae_config.yml")
    torch.multiprocessing.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
