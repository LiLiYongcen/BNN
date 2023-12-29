import os
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化图像
])
    

class SimpleCIFAR100(Dataset):
    def __init__(self, dataroot: str):
        super(SimpleCIFAR100, self).__init__()
        self.dataroot = dataroot
        data_path = os.path.join(dataroot, 'simple_cifar100', 'simple_cifar100.pt')
        if not os.path.exists(data_path):
            create_new_dataset(dataroot)
        self.data, self.labels = torch.load(data_path)
        self.transform = transform
        
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        return img, label
        
    def __len__(self):
        return len(self.data)


def create_new_dataset(dataroot: str):
    # 加载CIFAR-100训练集
    cifar100_path = os.path.join(dataroot, 'cifar100')
    cifar_trainset = datasets.CIFAR100(cifar100_path, train=True, download=True, transform=transform)
    cifar_loader = DataLoader(cifar_trainset, batch_size=1, shuffle=True)

    # 统计每个类别的样本数
    num_classes = 100
    samples_per_class = 50
    class_counts = [0] * num_classes

    # 用于存储新数据集的列表
    new_data = []
    new_labels = []

    # 随机抽取每个类别的样本
    for img, label in cifar_loader:
        label = label.item()
        img = img.squeeze(0)
        if class_counts[label] < samples_per_class:
            new_data.append(img)
            new_labels.append(label)
            class_counts[label] += 1

    # 打印新数据集的形状和类别分布
    # new_data_np = np.array(new_data)
    # new_labels_np = np.array(new_labels)
    # print("New dataset shape:", new_data_np.shape)
    # unique_labels, counts = np.unique(new_labels_np, return_counts=True)
    # print("Class distribution in new dataset:")
    # for label, count in zip(unique_labels, counts):
    #     print("Class", label, ":", count, "samples")

    # 保存新数据集
    simple_cifar100_dir = os.path.join(dataroot, 'simple_cifar100')
    if not os.path.exists(simple_cifar100_dir):
        os.mkdir(simple_cifar100_dir)
    simple_cifar100_path = os.path.join(dataroot, 'simple_cifar100', 'simple_cifar100.pt')
    torch.save((new_data, new_labels), simple_cifar100_path)