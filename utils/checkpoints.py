import torch


def save(model_state_dict, optimizer_state_dict, epoch, loss, out_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss,
    }, out_path)


def load(path):
    checkpoint = torch.load(path)
    return checkpoint['model_state_dict'], checkpoint['optimizer_state_dict'], checkpoint['epoch'], checkpoint['loss']
