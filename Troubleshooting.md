### Issue1:
Traceback (most recent call last):
  File "/Users/qiuhui/Workspace/Code/Lora/src/train.py", line 4, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'

### Solution1: 
/Users/qiuhui/anaconda3/envs/loratest/bin/pip install torch

Successfully installed MarkupSafe-3.0.3 filelock-3.20.0 fsspec-2025.12.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.6.1 sympy-1.14.0 torch-2.2.2 typing-extensions-4.15.0


/Users/qiuhui/anaconda3/envs/loratest/bin/pip install transformers
/Users/qiuhui/anaconda3/envs/loratest/bin/pip install peft
/Users/qiuhui/anaconda3/envs/loratest/bin/pip install datasets

### Issue2-Solution2:
If you have the SSL issue, pls try to run install_dependencies.sh.