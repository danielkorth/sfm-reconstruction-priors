import gc
import subprocess
import time
from functools import wraps
from typing import Any, Callable, Set, TypeVar

import torch

# Global set to track tensor IDs
_tensor_ids: Set[int] = set()


def track_tensor(tensor: torch.Tensor) -> None:
    """Track a tensor by its ID."""
    if tensor.is_cuda:
        _tensor_ids.add(id(tensor))


def cleanup_unreferenced_tensors():
    """Clean up tensors that were tracked but no longer have references."""
    print("[DEBUG] Cleaning up unreferenced tensors...")

    # Get all objects in memory
    objects = gc.get_objects()

    # Find tensors that are still referenced
    referenced_tensor_ids = set()
    for obj in objects:
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            referenced_tensor_ids.add(id(obj))

    # Find tensors that were tracked but are no longer referenced
    unreferenced_ids = _tensor_ids - referenced_tensor_ids

    # Clear the tracking set
    _tensor_ids.clear()

    # Print debug info
    print(f"[DEBUG] Found {len(unreferenced_ids)} unreferenced tensors")

    # Force garbage collection to clean up the unreferenced tensors
    gc.collect()
    torch.cuda.empty_cache()


def print_tensor_objects(gpu_only=True):
    """Print all tensor objects with detailed information.

    If gpu_only is True, only print tensors that are on a CUDA device.
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                # Get the actual tensor (handle both direct tensors and tensor attributes)
                tensor = obj if torch.is_tensor(obj) else obj.data

                # If gpu_only is True, skip tensors not on a CUDA device
                if gpu_only and not tensor.is_cuda:
                    continue

                # Calculate memory in MB
                memory_mb = tensor.element_size() * tensor.nelement() / 1024 / 1024

                print(f"\nObject Type: {type(obj)}")
                print(f"Shape: {tensor.size()}")
                print(f"Device: {tensor.device}")
                print(f"Dtype: {tensor.dtype}")
                print(f"Memory Usage: {memory_mb:.2f} MB")
                print(f"Requires Grad: {tensor.requires_grad}")
                if hasattr(obj, "name") and obj.name:
                    print(f"Name: {obj.name}")
                print("-" * 50)
        except Exception as e:
            pass


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Max GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Current GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Max GPU memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


def get_gpu_memory_usage() -> float:
    """Get the current GPU memory usage in MB using nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        used_memory, total_memory = map(int, result.stdout.strip().split(", "))
        free_memory = total_memory - used_memory
        return free_memory
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return 0


def clear_gpu_memory():
    """Clear GPU cache and empty all garbage."""
    gc.collect()  # Python garbage collection
    torch.cuda.empty_cache()  # Free up CUDA memory
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()  # Clean up IPC (Inter-Process Communication) resources
        print(f"GPU memory after clearing cache: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def with_gpu_memory_management(min_required_memory_mb: float = 8000):
    """Decorator for GPU memory management."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[DEBUG] Starting memory management for {func.__name__}")
            print(f"[DEBUG] Required memory: {min_required_memory_mb} MB")

            max_attempts = 10
            wait_time = 30  # seconds

            for attempt in range(max_attempts):
                print(f"\n[DEBUG] Attempt {attempt + 1}/{max_attempts}")
                free_memory = get_gpu_memory_usage()

                if free_memory >= min_required_memory_mb:
                    print("[DEBUG] Enough memory available, executing function")
                    try:
                        result = func(*args, **kwargs)
                        print("[DEBUG] Function execution completed")
                        return result
                    except RuntimeError as e:
                        import sys

                        # gonna commit suicide
                        sys.exit(1)
                else:
                    print(f"[DEBUG] Not enough memory, waiting {wait_time} seconds")
                    time.sleep(wait_time)

            print(
                "[DEBUG] Failed to acquire sufficient memory after maximum attempts. Gracefully ending process..."
            )
            clear_gpu_memory()
            return None  # Return None to indicate graceful termination

        return wrapper

    return decorator
