"""工具函数"""
import json
import yaml
import os
from pathlib import Path


def load_config(config_path="config/config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_sample_data(output_path="data/train.json", num_samples=100):
    """创建示例训练数据
    
    这里创建一个简单的问答数据集作为示例
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 示例数据：简单的问答对
    sample_data = []
    
    # 添加一些示例对话
    examples = [
        # {"input": "什么是机器学习？", "output": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进性能，而无需明确编程。"},
        # {"input": "什么是深度学习？", "output": "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂表示。"},
        # {"input": "什么是 LoRA？", "output": "LoRA（Low-Rank Adaptation）是一种参数高效的微调方法，通过低秩矩阵分解来减少可训练参数的数量。"},
        # {"input": "Python 是什么？", "output": "Python 是一种高级编程语言，以其简洁的语法和强大的功能而闻名，广泛用于数据科学、机器学习和 Web 开发。"},
        # {"input": "什么是 Transformer？", "output": "Transformer 是一种基于注意力机制的神经网络架构，是现代大语言模型的基础。"},
        {"input": "What is machine learning?","output": "Machine learning is a branch of artificial intelligence that enables computers to learn from data and improve performance without explicit programming."},
        {"input": "What is deep learning?","output": "Deep learning is a subset of machine learning that uses multi-layer neural networks to learn complex representations of data."},
        {"input": "What is LoRA?","output": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that reduces the number of trainable parameters through low-rank matrix factorization."},
        {"input": "What is Python?","output": "Python is a high-level programming language known for its concise syntax and powerful functions, widely used in data science, machine learning and web development."},
        {"input": "What is Transformer?","output": "Transformer is a neural network architecture based on the attention mechanism, which is the foundation of modern large language models."},
    ]
    
    # 重复生成数据以达到指定数量
    for i in range(num_samples):
        example = examples[i % len(examples)]
        sample_data.append({
            "text": f"问题：{example['input']}\n回答：{example['output']}"
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已创建 {num_samples} 条示例数据到 {output_path}")


def print_trainable_parameters(model):
    """打印可训练参数的数量"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"可训练参数: {trainable_params:,} || 总参数: {all_param:,} || 可训练比例: {100 * trainable_params / all_param:.2f}%")
