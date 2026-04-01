import torch
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, roc_auc_score
from utils import get_data_loaders, get_vgg16_model, get_device
import os

def plot_training_history(history_path='models/history.json'):
    """绘制训练历史曲线"""
    if not os.path.exists(history_path):
        print(f"未找到历史文件: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/training_curves.png')
    plt.close()
    print("训练曲线已保存到 images/training_curves.png")


def plot_confusion_matrix():
    """生成并绘制混淆矩阵"""
    device = get_device()
    model = get_vgg16_model()
    model.load_state_dict(torch.load('models/vgg16_cifar10.pth', weights_only=True))
    model.to(device)
    model.eval()
    
    _, testloader, classes = get_data_loaders(batch_size=128)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/confusion_matrix.png')
    plt.close()
    print("混淆矩阵已保存到 images/confusion_matrix.png")


def plot_class_metrics():
    """绘制各类别性能指标柱状图"""
    device = get_device()
    model = get_vgg16_model()
    model.load_state_dict(torch.load('models/vgg16_cifar10.pth', weights_only=True))
    model.to(device)
    model.eval()
    
    _, testloader, classes = get_data_loaders(batch_size=128)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 计算各类别指标
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(classes)), average=None, zero_division=0
    )
    
    # 绘制柱状图
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision')
    ax.bar(x, recall, width, label='Recall')
    ax.bar(x + width, f1, width, label='F1-Score')
    
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    os.makedirs('images', exist_ok=True)
    plt.tight_layout()
    plt.savefig('images/class_metrics.png')
    plt.close()
    print("各类别性能指标图已保存到 images/class_metrics.png")


def plot_auc_roc():
    """计算并可视化AUC-ROC指标"""
    device = get_device()
    model = get_vgg16_model()
    model.load_state_dict(torch.load('models/vgg16_cifar10.pth', weights_only=True))
    model.to(device)
    model.eval()
    
    _, testloader, classes = get_data_loaders(batch_size=128)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 计算macro AUC (one-vs-rest)
    try:
        auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        print(f"Macro AUC-ROC Score: {auc_score:.4f}")
    except Exception as e:
        print("AUC计算错误:", e)
        auc_score = 0.0
    
    # 简单AUC柱状图 (使用one-vs-rest per class)
    per_class_auc = []
    for i in range(len(classes)):
        try:
            y_binary = (all_labels == i).astype(int)
            class_auc = roc_auc_score(y_binary, all_probs[:, i])
            per_class_auc.append(class_auc)
        except:
            per_class_auc.append(0.5)
    
    # 绘制AUC柱状图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    plt.bar(x, per_class_auc, color='skyblue')
    plt.title('Per-Class AUC-ROC')
    plt.xlabel('Class')
    plt.ylabel('AUC Score')
    plt.xticks(x, classes, rotation=45)
    plt.axhline(y=auc_score, color='r', linestyle='--', label=f'Macro AUC: {auc_score:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('images', exist_ok=True)
    plt.tight_layout()
    plt.savefig('images/auc_roc.png')
    plt.close()
    print("AUC-ROC图已保存到 images/auc_roc.png")


if __name__ == "__main__":
    plot_training_history()
    plot_confusion_matrix()
    plot_class_metrics()
    plot_auc_roc()
    print("所有可视化完成！")