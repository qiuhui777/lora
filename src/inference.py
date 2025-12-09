"""LoRA 模型推理脚本"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import load_config


def generate_text(model, tokenizer, prompt, max_length=100, device="cpu"):
    """生成文本"""
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    print("=" * 50)
    print("LoRA 模型推理测试")
    print("=" * 50)
    
    # 加载配置
    config = load_config()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 检查模型是否存在
    model_path = config['training']['output_dir']
    print(f"\n正在加载模型: {model_path}")
    
    try:
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            cache_dir=config['model']['cache_dir'],
            torch_dtype=torch.float32
        )
        
        # 加载 LoRA 权重
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.to(device)
        model.eval()
        
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ 模型加载完成")
        
        # 测试提示词
        test_prompts = [
            "问题：什么是机器学习？\n回答：",
            "问题：什么是深度学习？\n回答：",
            "问题：什么是 LoRA？\n回答：",
        ]
        
        print("\n" + "=" * 50)
        print("开始生成测试")
        print("=" * 50 + "\n")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"测试 {i}:")
            print(f"输入: {prompt.strip()}")
            
            generated = generate_text(
                model, 
                tokenizer, 
                prompt, 
                max_length=150,
                device=device
            )
            
            print(f"输出: {generated}")
            print("-" * 50 + "\n")
        
        # 交互式测试
        print("\n进入交互模式（输入 'quit' 退出）:")
        print("-" * 50)
        
        while True:
            user_input = input("\n请输入问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("退出程序")
                break
            
            if not user_input:
                continue
            
            prompt = f"问题：{user_input}\n回答："
            generated = generate_text(
                model,
                tokenizer,
                prompt,
                max_length=150,
                device=device
            )
            
            print(f"\n生成结果:\n{generated}\n")
    
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示: 请先运行 'python src/train.py' 训练模型")


if __name__ == "__main__":
    main()
