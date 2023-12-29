import torch


def load_optimizer(cfg_training, model):
    if cfg_training['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg_training['lr'], 
            weight_decay=cfg_training['weight_decay'])
        
    elif cfg_training['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg_training['lr'], 
            momentum=cfg_training['momentum'], 
            weight_decay=cfg_training['weight_decay'])
    
    return optimizer