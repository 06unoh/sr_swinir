import torch
import torch.nn.functional as F
import math


def calc_psnr(outputs, hr):
    mse = F.mse_loss(outputs, hr, reduction='mean').item()
    if mse == 0:
        return 100 
    return 20 * math.log10(1.0 / math.sqrt(mse)) 

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total=0
    total_loss=0.0
    total_psnr=0.0
    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr=lr.to(device), hr.to(device)
            outputs=model(lr)
            
            loss=criterion(outputs, hr)
            total_loss+=loss.item()*lr.size(0)
            
            outputs_clamp=torch.clamp(outputs, 0.0, 1.0)
            psnr=calc_psnr(outputs_clamp, hr)
            total_psnr+=psnr*lr.size(0)            
            
            total+=hr.size(0)
            
    avg_loss=total_loss/total
    avg_psnr=total_psnr/total
    return avg_loss, avg_psnr