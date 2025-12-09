#!/bin/bash

echo "================================"
echo "安装 Python 依赖包"
echo "================================"

# 方法1: 使用国内镜像源（推荐）
echo ""
echo "尝试使用清华大学镜像源安装..."
pip install --trusted-host pypi.tuna.tsinghua.edu.cn \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 依赖安装成功！"
    exit 0
fi

# 如果失败，尝试方法2
echo ""
echo "尝试使用阿里云镜像源安装..."
pip install --trusted-host mirrors.aliyun.com \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 依赖安装成功！"
    exit 0
fi

# 如果还是失败，尝试方法3
echo ""
echo "尝试禁用 SSL 验证安装（不推荐，但可以解决证书问题）..."
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 依赖安装成功！"
else
    echo ""
    echo "✗ 安装失败，请手动尝试以下命令："
    echo "pip install --trusted-host pypi.tuna.tsinghua.edu.cn -i https://pypi.tuna.tsinghua.edu.cn/simple peft datasets accelerate pyyaml tqdm"
fi
