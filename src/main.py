import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "train.py"], check=True)
subprocess.run([sys.executable, "inference.py"], check=True)
subprocess.run([sys.executable, "convert.py"], check=True)
