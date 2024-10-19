import subprocess
import sys

train = False # установить True, если необходимо обучить модель с нуля

subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

if train:
  subprocess.run([sys.executable, "train.py"], check=True)
  
subprocess.run([sys.executable, "inference.py"], check=True)
subprocess.run([sys.executable, "convert.py"], check=True)
