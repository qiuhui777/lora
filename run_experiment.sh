#!/bin/bash

echo "================================"
echo "LoRA 微调实验启动脚本"
echo "================================"

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装 Python"
    exit 1
fi

echo ""
echo "步骤 1: 检查并安装依赖..."
echo "--------------------------------"
pip3 install -r requirements.txt

echo ""
echo "步骤 2: 开始训练模型..."
echo "--------------------------------"
python3 src/train.py

echo ""
echo "步骤 3: 测试模型..."
echo "--------------------------------"
python3 src/inference.py

echo ""
echo "================================"
echo "实验完成！"
echo "================================"
