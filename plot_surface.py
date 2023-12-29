import copy
import h5py
import torch
import time
import os
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from utils.general import set_seed, load_config, check_and_create_dir
from model.resnet.resnet import load_resnet
from model.vovnet.vovnet import load_vovnet
import utils.net_plotter as net_plotter
import utils.projection as proj
from utils.simple_cifar100 import SimpleCIFAR100
from torch.utils.data import DataLoader
import utils.plot_2D as plot_2D
from torch.autograd.variable import Variable


def get_unplotted_indices(vals, xcoordinates, ycoordinates=None):
    # Create a list of indices into the vectorizes vals
    inds = np.array(range(vals.size))
    
    inds = inds[vals.ravel() <= 0]

    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
    s1 = xcoord_mesh.ravel()[inds]
    s2 = ycoord_mesh.ravel()[inds]
    return inds, np.c_[s1,s2]


def eval_loss(net, criterion, loader, device):
    correct = 0
    total_loss = 0
    total = 0 # number of samples

    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            batch_size = inputs.size(0)
            total += batch_size
            inputs = Variable(inputs)
            targets = Variable(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets).sum().item()

    return total_loss/total, 100.*correct/total


def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, args):
    f = h5py.File(surf_file, 'r+')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        f[loss_key] = losses
        f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    inds, coords = get_unplotted_indices(losses, xcoordinates, ycoordinates)

    print('Computing %d values ...'% (len(inds)))

    criterion = nn.CrossEntropyLoss()

    # Loop over all uncalculated loss values
    for count, ind in tqdm(enumerate(inds), total=len(inds)):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args['dir_type'] == 'weights':
            net_plotter.set_weights(net, w, d, coord)
        elif args['dir_type'] == 'states':
            net_plotter.set_states(net, s, d, coord)

        loss, acc = eval_loss(net, criterion, dataloader, cfg['device'])

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Only the master node writes to the file - this avoids write conflicts
        f[loss_key][:] = losses
        f[acc_key][:] = accuracies
        f.flush()
        
    f.close()


if __name__ == '__main__':
    cfg_path = './config/plot_surface.yaml'
    
    cfg = load_config(cfg_path)
    set_seed(cfg['seed'])

    # 设置保存路径
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    save_dir = os.path.join(cfg['save_dir'], time_str)
    check_and_create_dir(save_dir)
    os.system('cp {} {}'.format(cfg_path, save_dir))
    
    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    cfg['xmin'], cfg['xmax'], cfg['xnum'] = [float(a) for a in cfg['x'].split(':')]
    cfg['ymin'], cfg['ymax'], cfg['ynum'] = [float(a) for a in cfg['y'].split(':')]
    
    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    if cfg['model_name'] == 'resnet':
        net = load_resnet(cfg)
    elif cfg['model_name'] == 'vovnet':
        net = load_vovnet(cfg)
    net.load_state_dict(torch.load(cfg['model_file'], map_location=cfg['device']))
    w = net_plotter.get_weights(net)
    s = copy.deepcopy(net.state_dict())
    
    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    dir_file = os.path.join(save_dir, cfg['dir_file'])
    net_plotter.setup_direction(cfg, dir_file, net)
    
    surf_file = os.path.join(save_dir, cfg['surf_file'])
    net_plotter.setup_surface(cfg, surf_file, dir_file)
    
    d = net_plotter.load_directions(dir_file)
    similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
    print('cosine similarity between x-axis and y-axis: %f' % similarity)
    
    #--------------------------------------------------------------------------
    # Setup dataloader
    #--------------------------------------------------------------------------
    dataset = SimpleCIFAR100(cfg['data_root'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    
    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    crunch(surf_file, net, w, s, d, dataloader, 'train_loss', 'train_acc', cfg)
    
    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    plot_2D.plot_2d_contour(surf_file, 'train_loss', cfg['vmin'], cfg['vmax'], cfg['vlevel'])