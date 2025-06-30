import torch

def save_checkpoint(model, epoch, optimizer, warmup_scheduler, plateau_scheduler, path):
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
        'plateau_scheduler_state_dict': plateau_scheduler.state_dict(),
    }, path)