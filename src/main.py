import torch
import subprocess
import sys


def install(package):
    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)


install("torch==2.4.0")
install("peft==0.13.2")
install("trl==0.11.1")
install("transformers==4.44.2")
install("bitsandbytes==0.44.1")
install("accelerate==0.34.2")
install("huggingface_hub==0.25.1")
install("unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git")

subprocess.run([sys.executable, "train.py"], check=True)

subprocess.run([sys.executable, "inference.py"], check=True)

subprocess.run([sys.executable, "convert.py"], check=True)


if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    install("packaging")
    install("ninja")
    install("einops")
    install("flash-attn>=2.6.3")
