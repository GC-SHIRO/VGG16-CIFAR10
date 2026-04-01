# VGG16 CIFAR-10 分类项目

## 介绍
本项目使用 PyTorch 实现 VGG16 模型对 CIFAR-10 数据集进行图像分类。CIFAR-10 包含 10 类 32x32 彩色图像，是经典的计算机视觉基准数据集。

## 安装依赖
```bash
pip install -r requirements.txt
```

## 项目结构
- `train.py`: 主要训练脚本
- `utils.py`: 工具函数（数据加载、模型等）
- `visualize.py`: 性能可视化脚本
- `demand.md`: 项目需求文档
- `images/`: 保存可视化结果
- `models/`: 保存训练好的模型

## 使用方法
1. 训练模型：
   ```bash
   python train.py
   ```

2. 可视化性能：
   ```bash
   python visualize.py
   ```

## 结果
训练后将在 `images/` 目录生成以下可视化图表：
- `training_curves.png`: 损失曲线和准确率曲线
- `confusion_matrix.png`: 混淆矩阵
- `class_metrics.png`: 各类别 Precision、Recall、F1-Score 柱状图
- `auc_roc.png`: 各类别及Macro AUC-ROC柱状图

## 环境要求
- Python 3.8+
- PyTorch with CUDA support (optional)