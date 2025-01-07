import json
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from model.autoencoder_kl import AutoencoderKL
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from ..stage1_dataset import MRGenDataset

@torch.no_grad()
def inference():
    VAE_PATH = '../train_vae/MRGen/'
    device = torch.device('cuda')

    test_CT_dataset = MRGenDataset(data_json_file='./radiopaedia_abdomen_ct_image_annotated.json', mode='test', stage='vae', modality='CT')
    test_MRI_dataset = MRGenDataset(data_json_file='./radiopaedia_abdomen_mri_image_annotated.json', mode='test', stage='vae', modality='MRI')
    val_dataset = ConcatDataset([test_CT_dataset, test_MRI_dataset])

    print(f"Validation dataset size: {len(val_dataset)}")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=128)

    vae = AutoencoderKL.from_pretrained(VAE_PATH, subfolder='vae')
    vae.to(device, dtype=torch.float32)
    vae.eval()
    
    l1_loss, mse_loss, psnr_values, ssim_values, results = [], [], [], [], []
    
    progress_bar = tqdm(val_dataloader, desc="Validating", leave=False)
    for i, data in enumerate(progress_bar):
        image = data['image'].to(device, dtype=torch.float32)
        b, c, h, w = image.shape
        
        # Repeat the channel dimension if the input is single-channel
        if vae.in_channels == 3 and c == 1:
            image = image.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            latents = vae.encode(image, return_dict=False)[0].sample()
            recons = vae.decode(latents)[0].clip(-1, 1)
            
            # Convert the reconstructed image back to single-channel if necessary
            if vae.in_channels == 3 and c == 1:
                image = image.mean(dim=1, keepdim=True)
                recons = recons.mean(dim=1, keepdim=True)
            
            l1_loss_temp = F.l1_loss(image, recons)
            mse_loss_temp = F.mse_loss(image, recons)
            
            l1_loss.append(l1_loss_temp.item())
            mse_loss.append(mse_loss_temp.item())
            
            image_np = image.cpu().numpy()
            recons_np = recons.cpu().numpy()
            
            # Ensure win_size is an odd value less than or equal to the smaller side of the images
            win_size = min(h, w) if min(h, w) % 2 != 0 else min(h, w) - 1
            win_size = max(3, win_size)  # Ensure win_size is at least 3
            
            psnr_temp = psnr(image_np, recons_np, data_range=2)
            # Calculate SSIM for each sample and each channel, then average
            ssim_temp = np.mean([
                ssim(image_np[i, j], recons_np[i, j], data_range=2, win_size=win_size)
                for i in range(b) for j in range(c)
            ])
            
            psnr_values.append(psnr_temp)
            ssim_values.append(ssim_temp)
            
            results.append({
                'index': i,
                'l1_loss': l1_loss_temp.item(),
                'mse_loss': mse_loss_temp.item(),
                'psnr': psnr_temp,
                'ssim': ssim_temp
            })
        
        # Calculate mean values
        mean_l1_loss = np.mean(l1_loss)
        mean_mse_loss = np.mean(mse_loss)
        mean_psnr = np.mean(psnr_values)
        mean_ssim = np.mean(ssim_values)
        
        # Print mean values
        print(f"Iteration {i}: mean_l1_loss={mean_l1_loss}, mean_mse_loss={mean_mse_loss}, mean_psnr={mean_psnr}, mean_ssim={mean_ssim}")
        
        # Save results every 50 iterations
        if (i + 1) % 10 == 0:
            with open('./vae_results.json', 'w') as f:
                json.dump(results, f, indent=4)
    
    # Save final results
    with open('./vae_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    inference()