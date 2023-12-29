import yaml
import random
import numpy as np
import torch
import os


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def check_and_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)