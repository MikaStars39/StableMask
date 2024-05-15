import importlib
import torch

def check_availability(package_name):
    try:
        importlib.import_module(package_name)
        print(f"{package_name} is available.")
    except ImportError:
        print(f"{package_name} is not available.")

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_availability('transformers')
    check_availability('torch')
    check_availability('pytorch_lightning')
    check_cuda()
