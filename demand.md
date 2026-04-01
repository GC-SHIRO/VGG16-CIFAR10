# VGG16 图像分类项目需求

## 项目目标
使用 VGG16 模型完成一个经典但非过于简单的图像分类任务。选择 CIFAR-10 数据集（10 类彩色图像），避免使用 MNIST 等过于简单的任务。

## 技术要求
- **框架**：使用 PyTorch 实现
- **模型**：基于 torchvision 的 VGG16 预训练模型，进行迁移学习或微调
- **数据集**：CIFAR-10
  - 自动下载并使用 torchvision.datasets
- **数据处理**：
  - 数据增强（RandomCrop, RandomHorizontalFlip 等）
  - 标准化处理
- **训练配置**：
  - 实现完整的训练循环
  - 支持 GPU 加速（如果可用）
  - 记录训练损失和准确率
- **评估**：
  - 在测试集上评估模型性能
  - 计算整体准确率、各类别准确率

## 性能评价可视化要求
- 训练过程中绘制 Loss 曲线和 Accuracy 曲线
- 混淆矩阵可视化
- 使用 matplotlib 实现所有可视化
- 保存可视化结果到 images/ 目录

## 项目结构
```
.
├── demand.md
├── requirements.txt
├── train.py
├── utils.py
├── visualize.py
├── images/
└── models/
```

## 交付物
- 可运行的 train.py（训练并保存模型）
- 可运行的 visualize.py（生成性能图表）
- 完整的 README.md 说明如何运行

请确保代码清晰、可复现，并添加适当的注释。
