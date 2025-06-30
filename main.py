from datasets.dataloader import get_dataloader
from models.swinir import SwinIR
from utils.evaluate import evaluate
from utils.test import test
from utils.train import train
from utils.visualize import visualize_prediction
from utils.data_utils import save_checkpoint

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


def train_loop(model, trainloader, valloader, testloader, criterion, optimizer, warmup_scheduler, plateau_scheduler, save_path, device):
    best_loss=float('inf')
    early_stop_counter=0
    patience=10
    
    for epoch in range(100):
        train_loss=train(model, trainloader, criterion, optimizer, device)
        val_loss, val_psnr=evaluate(model, valloader, criterion, device)
        
        if epoch<5:
            warmup_scheduler.step()
            current_lr=warmup_scheduler.get_last_lr()[0]
        
        else:
            plateau_scheduler.step(val_loss)
            current_lr=optimizer.param_groups[0]['lr']        
        
        print(f'Epoch: {epoch+1}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}dB')

        if epoch > 5:
            if val_loss<best_loss:
                best_loss=val_loss
                early_stop_counter=0
                save_checkpoint(model, epoch, optimizer, warmup_scheduler, plateau_scheduler, save_path)
                print(f'Save Best Model {epoch+1}')
            
            else:
                early_stop_counter+=1
                
            if early_stop_counter>=patience:
                print(f'Early Stop {epoch+1}')
                break   
    
    model.load_state_dict(torch.load(save_path)['model_state_dict'])            
    test(model, testloader, criterion, device)
    visualize_prediction(model, testloader, 5, device)
    
def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SwinIR().to(device)
    criterion=nn.L1Loss()
    optimizer=optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    
    warmup_scheduler=LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch+1)/ 5 if epoch<5 else 1.0)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    save_path = './checkpoints/swinir_best.pth'   
    
    trainloader, valloader, testloader=get_dataloader()
    
    train_loop(model, trainloader, valloader, testloader, criterion, optimizer, warmup_scheduler, plateau_scheduler, save_path, device)

if __name__=='__main__':
    main()