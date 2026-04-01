import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
import os
from utils import get_data_loaders, get_vgg16_model, get_device, save_model

def train_model(epochs=10, batch_size=128, lr=0.001, save_history=True):
    device = get_device()
    print(f"使用设备: {device}")
    
    # 获取数据加载器
    trainloader, testloader, classes = get_data_loaders(batch_size=batch_size)
    print(f"类别: {classes}")
    
    # 获取模型
    model = get_vgg16_model(fine_tune=True).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 记录历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = train_loss / len(trainloader)
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(testloader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss = val_loss / len(testloader)
        val_acc = 100. * correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # 保存模型
    save_model(model)
    
    # 保存训练历史
    if save_history:
        os.makedirs('models', exist_ok=True)
        with open('models/history.json', 'w') as f:
            json.dump(history, f)
        print("训练历史已保存到 models/history.json")
    
    return model, history


if __name__ == "__main__":
    # 为了演示，设置较少的epochs，实际使用可增加
    model, history = train_model(epochs=5, batch_size=128, lr=0.01)
    print("训练完成！")