import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def test(model, dataloader, criterion, device):
    model.eval()
    total=0
    total_loss=0.0
    total_psnr=0.0
    total_ssim=0.0
    
    with torch.no_grad():
        for lr_batch, hr_batch in dataloader:
            lr_batch, hr_batch=lr_batch.to(device), hr_batch.to(device)
            sr_batch=model(lr_batch)
            
            loss=criterion(sr_batch, hr_batch)
            total_loss+=loss.item()*lr_batch.size(0)
            total+=hr_batch.size(0)
            
            sr_imgs=sr_batch.clamp(0, 1).cpu().numpy()
            hr_imgs=hr_batch.clamp(0, 1).cpu().numpy()
            
            for sr, hr in zip(sr_imgs, hr_imgs):
                sr=np.transpose(sr, (1, 2, 0))
                hr=np.transpose(hr, (1, 2, 0))
                
                psnr_val=psnr(hr, sr, data_range=1.0)
                
                sr_h, sr_w, _=sr.shape
                min_dim=min(sr_h, sr_w)
                win_size=min(7, min_dim if min_dim%2==1 else min_dim-1)
                    
                if win_size>=3:
                    ssim_val=ssim(hr, sr, data_range=1.0, channel_axis=-1, win_size=win_size)
                
                else:
                    ssim_val=0
                
                total_psnr+=psnr_val
                total_ssim+=ssim_val
                
    print(f'L1 LOSS: {total_loss/total:.4f}, PSNR: {total_psnr/total:.2f}dB, SSIM: {total_ssim/total:.4f}')
            
        