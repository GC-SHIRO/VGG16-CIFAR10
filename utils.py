import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

def get_data_loaders(batch_size=128, num_workers=2):
    """
    获取CIFAR-10数据加载器，包含训练数据增强
    """
    # CIFAR-10标准化参数
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # 下载并加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    testloader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return trainloader, testloader, trainset.classes


def get_vgg16_model(fine_tune=True):
    """
    获取修改后的VGG16模型，适应CIFAR-10的10类输出
    """
    # 加载预训练VGG16模型
    model = torchvision.models.vgg16(pretrained=True)
    
    # 修改分类器以适应10个类别
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 10)
    
    if not fine_tune:
        # 如果不微调，冻结特征提取层
        for param in model.features.parameters():
            param.requires_grad = False
    
    return model


def get_device():
    """获取可用设备"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_model(model, path='models/vgg16_cifar10.pth'):
    """保存模型"""
    import os
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")