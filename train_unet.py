import os
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Optional
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.cuda.amp import autocast
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModel, CLIPTextModel
from diffusers.optimization import get_scheduler
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available

from stage1_dataset import MRGenDataset
from model.autoencoder_kl import AutoencoderKL
from model.pipeline import StableDiffusionPipeline
from model.unet_2d_condition import UNet2DConditionModel
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
        os.makedirs(self.logdir)
        
    def log_sample_images(self, batch, pipeline: StableDiffusionPipeline, device: torch.device, step: int):
        sample_seeds = torch.randint(0, 100000, (self.num_sample_per_prompt,))
        sample_seeds = sorted(sample_seeds.numpy().tolist())
        self.sample_seeds = sample_seeds
        self.prompts = batch["prompt"]
        for idx, prompt in enumerate(tqdm(self.prompts, desc="Generating sample images")):
            image = batch["image"][idx, :, :, :].unsqueeze(0)
            image = image.to(device=device)
            generator = [torch.Generator(device=device).manual_seed(seed) for seed in self.sample_seeds]
            sequence = pipeline(
                prompt,
                height=image.shape[2],
                width=image.shape[3],
                generator=generator,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                num_images_per_prompt=self.num_sample_per_prompt,
            ).images

            image = (image + 1.) / 2. # for visualization
            image = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            cv2.imwrite(os.path.join(self.logdir, f"{step}_{idx}.png"), image[:, :, ::-1] * 255)
            with open(os.path.join(self.logdir, f"{step}_{idx}" + '.txt'), 'a') as f:
                f.write(batch['prompt'][idx])
            for i, img in enumerate(sequence):
                img[0].save(os.path.join(self.logdir, f"{step}_{idx}_{sample_seeds[i]}_output.png"))
            
def train(
    data_ct_json_file: str,
    data_mri_json_file: str,
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

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))

    if seed is not None:
        set_seed(seed)

    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False) # CLIP Tokenizer
    # text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder") # CLIP Text Encoder
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_BiomedCLIP", use_fast=False) # BiomedCLIP Tokenizer
    text_encoder = AutoModel.from_pretrained(os.path.join(pretrained_model_path, 'text_encoder_BiomedCLIP'), trust_remote_code=True) # BiomedCLIP Text Encoder
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    # unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet") # if loading from pretrained checkpoint
    unet = UNet2DConditionModel.from_config(pretrained_model_path, subfolder="unet") # if training from scratch
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    pipeline = StableDiffusionPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
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
    unet.requires_grad_(True)
    
    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)

    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    
    train_MRI_dataset = MRGenDataset(data_json_file=data_mri_json_file, mode='train', stage='unet', modality='MRI')
    test_MRI_dataset = MRGenDataset(data_json_file=data_mri_json_file, mode='test', stage='unet', modality='MRI')
    
    train_dataset = train_MRI_dataset
    val_dataset = test_MRI_dataset
    
    print(train_dataset.__len__())
    print(val_dataset.__len__())
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, pin_memory=True, num_workers=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, pin_memory=True, num_workers=32, shuffle=False)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=train_steps * gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
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
        text_encoder.eval()
        unet.train()
        
        image = batch["image"].to(dtype=weight_dtype)
        prompt = batch["prompt"]
        
        text_inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=77, return_tensors="pt")
        text_input_ids = text_inputs.input_ids.to(accelerator.device)
        encoder_hidden_states = text_encoder.text_model(text_input_ids)[0] # B * L * 768

        b, c, h, w = image.shape
        latents = vae.encode(image).latent_dist.sample() * vae.scaling_factor
        noise = torch.randn_like(latents) # [-1, 1]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=latents.device).long()
        noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps)

        model_pred = unet(noisy_latent, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            step += 1
            if accelerator.is_main_process:
                if validation_sample_logger is not None and step % validation_steps == 0:
                    unet.eval()
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
                    pipeline_save = StableDiffusionPipeline(
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        scheduler=scheduler,
                        unet=accelerator.unwrap_model(unet),
                    )
                    checkpoint_save_path = os.path.join(logdir, f"checkpoint_latest")
                    pipeline_save.save_pretrained(checkpoint_save_path)
                
                if step % checkpointing_steps == 0:
                    pipeline_save = StableDiffusionPipeline(
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        scheduler=scheduler,
                        unet=accelerator.unwrap_model(unet),
                    )
                    checkpoint_save_path = os.path.join(logdir, f"checkpoint_{step}")
                    pipeline_save.save_pretrained(checkpoint_save_path)

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=step)
    accelerator.end_training()

if __name__ == "__main__":
    config = "./training_configs/train_unet_config.yml"
    train(**OmegaConf.load(config))