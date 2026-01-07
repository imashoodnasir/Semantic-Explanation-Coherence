import os
from typing import Optional

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def join(*parts: str) -> str:
    return os.path.join(*parts)

def file_exists(path: str) -> bool:
    return os.path.exists(path)
