# 快速开始指南

## 1. 环境准备

### 安装 Python 依赖
```bash
pip install -r requirements.txt
```

如果下载速度慢，可以使用国内镜像：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 运行实验

### 方法一：使用一键脚本（推荐）
```bash
./run_experiment.sh
```

### 方法二：分步执行

#### 步骤 1: 训练模型
```bash
python src/train.py
```

训练过程会：
- 自动下载 GPT-2 模型（约 500MB）
- 生成示例训练数据
- 使用 LoRA 进行微调
- 保存模型到 `models/lora_model/`

#### 步骤 2: 测试模型
```bash
python src/inference.py
```

测试过程会：
- 加载训练好的模型
- 运行预设的测试样例
- 进入交互模式，可以输入自定义问题

## 3. 配置调整

编辑 `config/config.yaml` 可以调整：

### LoRA 参数
- `r`: LoRA rank，越大模型容量越大（建议 4-16）
- `lora_alpha`: 缩放因子（通常是 r 的 2 倍）
- `lora_dropout`: Dropout 比例

### 训练参数
- `num_epochs`: 训练轮数
- `batch_size`: 批次大小（内存不足可以减小）
- `learning_rate`: 学习率
- `max_length`: 最大序列长度

## 4. 使用自己的数据

### 数据格式
在 `data/train.json` 中准备数据，格式如下：
```json
[
  {
    "text": "问题：你的问题\n回答：对应的答案"
  },
  {
    "text": "问题：另一个问题\n回答：另一个答案"
  }
]
```

### 修改配置
在 `config/config.yaml` 中设置：
```yaml
data:
  train_file: "./data/train.json"
  max_samples: 1000  # 根据需要调整
```

## 5. 性能优化建议

### 如果内存不足
- 减小 `batch_size`（如改为 2 或 1）
- 减小 `max_length`（如改为 64）
- 减小 `max_samples`（减少训练数据量）

### 如果有 GPU
训练会自动使用 GPU，速度会快很多

### 如果训练太慢
- 减少 `num_epochs`
- 减少训练数据量
- 使用更小的 LoRA rank

## 6. 常见问题

### Q: 下载模型失败？
A: 可能是网络问题，可以：
- 使用代理
- 手动下载模型到 `models/cache/` 目录

### Q: 内存不足？
A: 调整配置文件中的 `batch_size` 和 `max_length`

### Q: 训练效果不好？
A: 可以尝试：
- 增加训练数据量
- 增加训练轮数
- 调整学习率
- 增大 LoRA rank

## 7. 项目结构说明

```
.
├── config/
│   └── config.yaml          # 配置文件
├── src/
│   ├── train.py            # 训练脚本
│   ├── inference.py        # 推理脚本
│   └── utils.py            # 工具函数
├── data/                   # 数据目录（自动生成）
├── models/                 # 模型目录（自动生成）
├── requirements.txt        # Python 依赖
├── run_experiment.sh       # 一键运行脚本
└── README.md              # 项目说明
```

## 8. 下一步

- 尝试使用自己的数据集
- 调整 LoRA 参数观察效果
- 尝试不同的基础模型
- 添加评估指标
