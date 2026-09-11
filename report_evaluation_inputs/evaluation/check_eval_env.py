#!/usr/bin/env python3
"""Fail-fast environment checks for the RaTEScore GPU job."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys


REQUIRED_PACKAGES = (
    "radeval",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "accelerate",
    "numpy",
)


def version_for(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    print("Python executable:", sys.executable)
    print("Python version:", platform.python_version())

    failures = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"OK      {package} {version_for(package)}")
        except Exception as error:
            failures.append(package)
            print(f"FAILED  {package}: {type(error).__name__}: {error}")

    if failures:
        raise SystemExit("Missing or broken packages: " + ", ".join(failures))

    import torch

    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this check inside a GPU Slurm allocation.")

    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}:", torch.cuda.get_device_name(index))

    from radeval import RadEval

    RadEval(metrics=["ratescore"], per_sample=True)
    print("RaTEScore evaluator initialized successfully")


if __name__ == "__main__":
    main()
