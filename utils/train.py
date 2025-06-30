from torch import nn

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total=0
    total_loss=0.0
    for lr, hr in dataloader:
        lr, hr=lr.to(device), hr.to(device)
        
        outputs=model(lr)
        loss=criterion(outputs, hr)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()        
        total_loss+=loss.item()*lr.size(0)
        total+=hr.size(0)
    avg_loss=total_loss/total
    return avg_loss
        
        
    