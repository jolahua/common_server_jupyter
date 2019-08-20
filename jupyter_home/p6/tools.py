#!/us/bin/python
# -*- coding:utf-8 -*-


import json

import torch
from torchvision import datasets, transforms


def load_datas(folder):
    train_dir = folder + '/train'
    valid_dir = folder + '/valid'
    test_dir = folder + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),  # 随机旋转
                                       transforms.RandomResizedCrop(224),  # 裁剪为随机大小和高度比
                                       transforms.RandomHorizontalFlip(),  # 给定概率随机水平翻转
                                       transforms.ToTensor(),  # 转化为张量，即totch可读类型
                                       transforms.Normalize([0.485, 0.456, 0.406],  # 用平均值和标准差归一化张量图像
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = test_transforms

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    # 图像数据加载器

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    # 数据加载器  组合数据集和采样器
    return trainloaders, validloaders, testloaders

def get_class_to_ids(file):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name