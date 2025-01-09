import os
import cv2
import torch
import bitsandbytes as bnb
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import Dict, Optional
from torch.cuda.amp import autocast
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from transformers import AutoTokenizer, AutoModel
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

from stage2_dataset import CHAOS_MRI_Dataset
from model.controlnet import ControlNetModel
from model.autoencoder_kl import AutoencoderKL
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline_controlnet import StableDiffusionControlNetPipeline
from utils.util import get_function_args, get_time_string

logger = get_logger(__name__)

class SampleLogger:
    def __init__(
        self,
        logdir: str,
        subdir: str = "sample",
        num_samples_per_prompt: int = 1,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.0,
    ) -> None:
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.num_sample_per_prompt = num_samples_per_prompt
        self.logdir = os.path.join(logdir, subdir)
        os.makedirs(self.logdir, exist_ok=True)
        
    def log_sample_images(self, batch, pipeline: StableDiffusionControlNetPipeline, device: torch.device, step: int):
        sample_seeds = sorted(torch.randint(0, 100000, (self.num_sample_per_prompt,)).tolist())
        self.sample_seeds = sample_seeds
        self.prompts = batch["prompt"]
        for idx, prompt in enumerate(tqdm(self.prompts, desc="Generating sample images")):
            image = batch["image"][idx, :, :, :].unsqueeze(0).to(device=device)
            # mask = batch["mask"][idx, :, :, :].unsqueeze(0).to(device=device)
            mask = batch["merged_mask"][idx, :, :, :].unsqueeze(0).to(device=device)
            merged_mask = batch["merged_mask"][idx, :, :, :].unsqueeze(0)
            generator = [torch.Generator(device=device).manual_seed(seed) for seed in self.sample_seeds]
            
            sequence = pipeline(
                # prompt=[", ".join(['T1 in-phase Abdomen MRI, fat high signal, muscle intermediate signal, water low signal, fat bright', batch['region'][0], batch['organs'][0]])],
                prompt=[", ".join(['T1 in-phase Abdomen MRI, fat high signal, muscle intermediate signal, water low signal, fat bright', batch['region'][0], batch['label'][0]])],
                image=mask,
                height=image.shape[2],
                width=image.shape[3],
                generator=generator,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                num_images_per_prompt=self.num_sample_per_prompt,
            ).images
            
            sequence_T2_SPIR = pipeline(
                # prompt=[", ".join(['T2 SPIR Abdomen MRI, fat low signal, muscle low signal, water high signal, fat dark, water bright', batch['region'][0], batch['organs'][0]])],
                prompt=[", ".join(['T2 SPIR Abdomen MRI, fat low signal, muscle low signal, water high signal, fat dark, water bright', batch['region'][0], batch['label'][0]])],
                image=mask,
                height=image.shape[2],
                width=image.shape[3],
                generator=generator,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                num_images_per_prompt=self.num_sample_per_prompt,
            ).images

            image = (image + 1.) / 2. # for visualization
            image = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            merged_mask = (merged_mask + 1.) / 2. # for visualization
            merged_mask = merged_mask.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            cv2.imwrite(os.path.join(self.logdir, f"{step}_{idx}.png"), image[:, :, ::-1] * 255)
            cv2.imwrite(os.path.join(self.logdir, f"{step}_{idx}_mask.png"), merged_mask[:, :, ::-1] * 255)
            
            with open(os.path.join(self.logdir, f"{step}_{idx}" + '.txt'), 'a') as f:
                f.write(f"{batch['prompt'][idx]}\n{batch['image_path'][idx]}\n")
            for i, img in enumerate(sequence):
                img.save(os.path.join(self.logdir, f"{step}_{idx}_{sample_seeds[i]}_output_T1-in-phase.png"))
            for i, img in enumerate(sequence_T2_SPIR):
                img.save(os.path.join(self.logdir, f"{step}_{idx}_{sample_seeds[i]}_output_T2_SPIR.png"))
            
def train(
    pretrained_model_path: str,
    logdir: str,
    train_steps: int = 5000,
    validation_steps: int = 1000,
    validation_sample_logger: Optional[Dict] = None,
    gradient_accumulation_steps: int = 1, # important hyper-parameter
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    train_batch_size: int = 1,
    val_batch_size: int = 1,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",  # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    lr_warmup_steps: int = 0,
    use_8bit_adam: bool = True,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    checkpointing_steps: int = 10000,
):
    args = get_function_args()
    logdir += f"_{get_time_string()}"

    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision=mixed_precision)
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))

    if seed is not None:
        set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_BiomedCLIP", use_fast=False)
    text_encoder = AutoModel.from_pretrained(os.path.join(pretrained_model_path, 'text_encoder_BiomedCLIP'), trust_remote_code=True)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    # controlnet = ControlNetModel.from_pretrained(pretrained_model_path, subfolder="controlnet") # load pretrained controlnet weights
    controlnet = ControlNetModel.from_unet(unet) # load pretrained unet weights to controlnet as initialization
    
    pipeline = StableDiffusionControlNetPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning("Could not enable memory efficient attention. Make sure xformers is installed correctly and a GPU is available: {e}")
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(True)

    if scale_lr:
        learning_rate = learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
    
    optimizer_class = torch.optim.AdamW if not use_8bit_adam else bnb.optim.AdamW8bit
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(params_to_optimize, lr=learning_rate, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon)

    # Pay attention to the target_modality
    train_dataset = CHAOS_MRI_Dataset(mode='train', target_modality='T2-SPIR')
    val_dataset = CHAOS_MRI_Dataset(mode='test', target_modality='T2-SPIR')
    # train_dataset = CHAOS_MRI_Dataset(mode='train', target_modality='T1-InPhase')
    # val_dataset = CHAOS_MRI_Dataset(mode='test', target_modality='T1-InPhase')
    # train_dataset = CHAOS_MRI_Dataset(mode='train', target_modality='T1-OutofPhase')
    # val_dataset = CHAOS_MRI_Dataset(mode='test', target_modality='T1-OutofPhase')
    print(train_dataset.__len__())
    print(val_dataset.__len__())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, num_workers=32, pin_memory=True, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, num_workers=32, pin_memory=True, shuffle=False)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=train_steps * gradient_accumulation_steps,
    )

    unet, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, controlnet, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("MRGen")
    step = 0

    if validation_sample_logger is not None and accelerator.is_main_process:
        validation_sample_logger = SampleLogger(**validation_sample_logger, logdir=logdir)

    progress_bar = tqdm(range(step, train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)
    val_data_yielder = make_data_yielder(val_dataloader)

    while step < train_steps:
        batch = next(train_data_yielder)
        
        vae.eval()
        unet.eval()
        text_encoder.eval()
        controlnet.train()
        
        image = batch["image"].to(dtype=weight_dtype)
        # mask = batch["mask"].to(dtype=weight_dtype)
        mask = batch["merged_mask"].to(dtype=weight_dtype)
        prompt = batch["prompt"]
        
        b, c, h, w = image.shape

        text_inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=77, return_tensors="pt")
        text_input_ids = text_inputs.input_ids.to(accelerator.device)
        latents = vae.encode(image).latent_dist.sample() * vae.scaling_factor
        noise = torch.randn_like(latents) # [-1, 1]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=latents.device).long()
        noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps)
        
        encoder_hidden_states = text_encoder.text_model(text_input_ids)[0] # B * L * 768
        down_block_res_samples, mid_block_res_sample = controlnet(noisy_latent, timesteps, encoder_hidden_states=encoder_hidden_states, controlnet_cond=mask, return_dict=False)
        
        model_pred = unet(
                    noisy_latent, timesteps, encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False)[0]
        
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(controlnet.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            step += 1
            if accelerator.is_main_process:
                if validation_sample_logger is not None and step % validation_steps == 0:
                    controlnet.eval()
                    val_batch = next(val_data_yielder)
                    with autocast():
                        validation_sample_logger.log_sample_images(
                            batch = val_batch,
                            pipeline=pipeline,
                            device=accelerator.device,
                            step=step,
                        )
                # latest checkpoint
                if step % (checkpointing_steps // 10) == 0:
                    pipeline_save = StableDiffusionControlNetPipeline(
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        scheduler=scheduler,
                        unet=accelerator.unwrap_model(unet),
                        controlnet=accelerator.unwrap_model(controlnet),
                    )
                    checkpoint_save_path = os.path.join(logdir, f"checkpoint_latest")
                    pipeline_save.save_pretrained(checkpoint_save_path)
                
                if step % checkpointing_steps == 0:
                    pipeline_save = StableDiffusionControlNetPipeline(
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        scheduler=scheduler,
                        unet=accelerator.unwrap_model(unet),
                        controlnet=accelerator.unwrap_model(controlnet),
                    )
                    checkpoint_save_path = os.path.join(logdir, f"checkpoint_{step}")
                    pipeline_save.save_pretrained(checkpoint_save_path)

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=step)
    accelerator.end_training()

if __name__ == "__main__":
    config = "./training_configs/train_controlnet_config.yml"
    train(**OmegaConf.load(config))