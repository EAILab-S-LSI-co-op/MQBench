import torch

@torch.no_grad()
def evaluate(val_loader, model, device=None):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
        
    model.eval()
    correct = 0
    for i, (x, y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        
        # Calculate accuracy
        _, predicted = torch.max(output, dim=1)
        correct += (predicted == y).sum().item()
        
    accuracy = correct / len(val_loader.dataset)
    return accuracy


def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    return cali_data