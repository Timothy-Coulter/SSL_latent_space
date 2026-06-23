"""Environment health check."""

import sys


def check_environment() -> bool:
    """Check if environment is properly configured."""
    checks: list[tuple[str, str, bool]] = []

    # Python version
    checks.append(("Python version", f"{sys.version_info.major}.{sys.version_info.minor}", True))

    # PyTorch
    try:
        import torch

        checks.append(("PyTorch", torch.__version__, True))
        checks.append(("CUDA available", str(torch.cuda.is_available()), torch.cuda.is_available()))
        if torch.cuda.is_available():
            try:
                # Access CUDA version
                cuda_version = torch.version.cuda
            except (AttributeError, RuntimeError):
                cuda_version = "UNKNOWN"
            checks.append(("CUDA version", str(cuda_version), True))
            checks.append(("GPU count", str(torch.cuda.device_count()), True))
    except ImportError:
        checks.append(("PyTorch", "NOT FOUND", False))

    # Print results
    print("\n=== Environment Health Check ===\n")
    for name, value, status in checks:
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}: {value}")

    return all(status for _, _, status in checks)


if __name__ == "__main__":
    sys.exit(0 if check_environment() else 1)
