import torch
import matplotlib.pyplot as plt

def visualize_prediction(model, dataloader, max_sample, device):
    model.eval()
    count=0
    
    with torch.no_grad():
        for lr, hr in dataloader:
            lr, hr=lr.to(device), hr.to(device)
            sr=model(lr)
            
            for i in range(lr.size(0)):
                if count >= max_sample:
                    return

                sr_img=sr[i].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                hr_img=hr[i].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                
                plt.figure(figsize=(8, 4))
                plt.suptitle(f"Sample {count+1}")
                
                plt.subplot(1, 2, 1)
                plt.imshow(hr_img)
                plt.title('Real')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(sr_img)
                plt.title('Prediction')
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
                
                count+=1
            
            