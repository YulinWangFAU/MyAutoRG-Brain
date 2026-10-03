import importlib
import os
import sys


MODULES = [
    "torch",
    "transformers",
    "batchgenerators",
    "SimpleITK",
    "skimage",
    "einops",
    "torchinfo",
]


def main():
    print("python:", sys.version.replace("\n", " "))
    print("cwd:", os.getcwd())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    for name in MODULES:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name}: {version}")

    import torch

    print("torch cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    print("gpu count:", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        mem_gb = props.total_memory / 1024**3
        print(f"gpu {idx}: {props.name}, {mem_gb:.1f} GB")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this inside a GPU Slurm job.")


if __name__ == "__main__":
    main()
