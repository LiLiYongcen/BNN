import os
import torch
import time
from tqdm import tqdm
from utils.metrics import accuracy_topk
from utils.general import load_config, set_seed, check_and_create_dir
from utils.dataset import load_dataset
from model.resnet.resnet import load_resnet
from utils.optimizer import load_optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    cfg_path = './config/resnet.yaml'
    
    # load config
    cfg = load_config(cfg_path)
    
    set_seed(cfg['seed'])

    # 设置保存路径
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    save_dir = os.path.join(cfg['save']['save_dir'], time_str)
    model_dir = os.path.join(save_dir, cfg['save']['model_dir'])
    check_and_create_dir(model_dir)
    log_dir = os.path.join(save_dir, cfg['save']['log_dir'])
    writer = SummaryWriter(log_dir=log_dir)
    os.system('cp {} {}'.format(cfg_path, save_dir))
    
    # load dataset
    cfg_training = cfg['training']
    
    train_dataset, test_dataset = load_dataset(cfg)
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg_training['batch_size'], 
                              shuffle=True, 
                              num_workers=cfg_training['num_workers'])
    
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg_training['batch_size'],
                             shuffle=False,
                             num_workers=cfg_training['num_workers'])
    
    DEVICE = cfg_training['device']
    # load model
    model = load_resnet(cfg).to(DEVICE)
    
    # loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = load_optimizer(cfg_training, model)
    
    # train
    for epoch in range(cfg_training['epochs']):
        # train
        print('Training... Epoch: [{}/{}]'.format(epoch + 1, cfg_training['epochs']))
        model.train()
        train_loss = 0
        train_idx = 0
        with tqdm(train_loader, desc='Epoch: [{}/{}]'.format(epoch + 1, cfg_training['epochs'])) as t:
            for data, target in t:
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_idx += 1
                
                t.set_postfix({'Average Loss': '{:.4f}'.format(train_loss / train_idx), 'Loss': '{:.4f}'.format(loss.item())})
            writer.add_scalar('train_loss', train_loss / train_idx, epoch + 1)
        
        # test
        print('Testing...')
        model.eval()
        test_loss = 0
        test_idx = 0
        
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with tqdm(test_loader, desc='Epoch: [{}/{}]'.format(epoch + 1, cfg_training['epochs'])) as t:
            for data, target in t:
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                test_idx += 1
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct_top1 += accuracy_topk(output, target, k=1)
                correct_top5 += accuracy_topk(output, target, k=5)
                
                t.set_postfix({'Average Loss': '{:.4f}'.format(test_loss / test_idx), 'Loss': '{:.4f}'.format(loss.item()), 'Accuracy Top1': '{:.4f}%'.format(100 * correct_top1 / total), 'Accuracy Top5': '{:.4f}%'.format(100 * correct_top5 / total)})
            writer.add_scalar('test_loss', test_loss / test_idx, epoch + 1)
            writer.add_scalar('test_accuracy_top1', correct_top1 / total, epoch + 1)
            writer.add_scalar('test_accuracy_top5', correct_top5 / total, epoch + 1)
        
        # save model
        if (epoch + 1) % cfg['save']['model_save_freq'] == 0:
            model_path = os.path.join(model_dir, 'model_{}.pth'.format(epoch + 1))
            torch.save(model.state_dict(), model_path)