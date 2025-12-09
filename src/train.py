"""LoRA 微调训练脚本"""
import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from utils import load_config, create_sample_data, print_trainable_parameters


def prepare_dataset(data_path, tokenizer, max_length=128, max_samples=None):
    """准备训练数据集"""
    print(f"正在加载数据: {data_path}")
    
    # 如果数据文件不存在，创建示例数据
    if not os.path.exists(data_path):
        print("数据文件不存在，正在创建示例数据...")
        create_sample_data(data_path, num_samples=max_samples or 100)
    
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    # 转换为 Dataset 对象
    dataset = Dataset.from_list(data)
    
    # Tokenize 数据
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"✓ 数据集准备完成，共 {len(tokenized_dataset)} 条样本")
    return tokenized_dataset


def main():
    print("=" * 50)
    print("LoRA 微调训练开始")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 加载 tokenizer 和模型
    print(f"\n正在加载模型: {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        cache_dir=config['model']['cache_dir']
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        cache_dir=config['model']['cache_dir'],
        torch_dtype=torch.float32  # 使用 float32 以兼容 CPU
    )

    print("✓ 模型加载完成")
    
    # 配置 LoRA
    print("\n配置 LoRA...")
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        bias=config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    print("✓ LoRA 配置完成")
    
    # 打印参数信息
    print("\n模型参数信息:")
    print_trainable_parameters(model)
    
    # 准备数据集
    train_dataset = prepare_dataset(
        config['data']['train_file'],
        tokenizer,
        max_length=config['training']['max_length'],
        max_samples=config['data'].get('max_samples')
    )
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        save_total_limit=2,
        fp16=False,  # CPU 不支持 fp16
        report_to="none",  # 不使用外部日志工具
        remove_unused_columns=False,
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型不使用 MLM
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50 + "\n")
    
    trainer.train()
    
    # 保存模型
    print("\n保存模型...")
    model.save_pretrained(config['training']['output_dir'])
    tokenizer.save_pretrained(config['training']['output_dir'])
    
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"模型已保存到: {config['training']['output_dir']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
