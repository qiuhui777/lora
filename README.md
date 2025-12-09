# LoRA 微调实验项目

这是一个轻量级的 LoRA (Low-Rank Adaptation) 微调实验项目，可以在笔记本电脑上运行。

## 项目结构

```
lora-finetuning/
├── data/                  # 训练数据目录
├── models/               # 模型保存目录
├── config/               # 配置文件
├── src/                  # 源代码
│   ├── train.py         # 训练脚本
│   ├── inference.py     # 推理脚本
│   └── utils.py         # 工具函数
├── requirements.txt      # 依赖包
└── README.md            # 项目说明
```

## 环境要求

- Python 3.8+
- 至少 8GB RAM
- 建议有 GPU，但 CPU 也可以运行

```
conda create -n loratest python=3.12
To activate this environment, use
    $ conda activate loratest
To deactivate an active environment, use
    $ conda deactivate
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 准备数据（示例数据会自动生成）

3. 开始训练：
```bash
python src/train.py
```

4. 测试模型：
```bash
python src/inference.py
```

## 配置说明

在 `config/config.yaml` 中可以调整：
- 模型参数
- LoRA 参数（rank, alpha）
- 训练超参数
- 数据路径

## 特点

- 使用小型模型（GPT-2 small）适合笔记本运行
- LoRA 技术大幅减少训练参数
- 支持 CPU 和 GPU 训练
- 包含完整的训练和推理流程

## 重新训练的方法
```
# 删除旧模型
rm -rf models/

# 重新训练
python src/train.py

# 测试
python src/inference.py
```