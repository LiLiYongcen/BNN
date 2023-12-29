import os
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# 定义数据增强的转换
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转角度范围为±10度
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化图像
])


def load_dataset(cfg: dict) -> (Dataset, Dataset):
    path = os.path.join(cfg['data_root'], cfg['dataset'])
    if cfg['dataset'] == 'cifar100':
        train_dataset = get_cifar100(path, train=True, download=True, transform=transform)
        test_dataset = get_cifar100(path, train=False, download=True, transform=transform)
    elif cfg['dataset'] == 'cifar10':
        train_dataset = get_cifar10(path, train=True, download=True, transform=transform)
        test_dataset = get_cifar10(path, train=False, download=True, transform=transform)
        
    return train_dataset, test_dataset


def get_cifar100(root, train=True, download=True, transform=None):
    return torchvision.datasets.CIFAR100(
        root=root, train=train, download=download, 
        transform=transform)


def get_cifar10(root, train=True, download=True, transform=None):
    return torchvision.datasets.CIFAR10(
        root=root, train=train, download=download, 
        transform=transform)
    
    
if __name__ == '__main__':
    train_dataset_100 = get_cifar100(root='./data/cifar100', train=True, download=True, transform=transform)
    test_dataset_100 = get_cifar100(root='./data/cifar100', train=False, download=True, transform=transform)
    
    train_dataset_10 = get_cifar10(root='./data/cifar10', train=True, download=True, transform=transform)
    test_dataset_10 = get_cifar10(root='./data/cifar10', train=False, download=True, transform=transform)
    pass