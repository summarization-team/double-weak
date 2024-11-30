# Description: This script verifies the setup of the environment, including the Python version and the installation of necessary dependencies.


import sys

def verify_python_version():
    if sys.version_info < (3, 9):
        print("Python 3.9 or higher is required.")
        sys.exit(1)
    else:
        print(f"Python version: {sys.version}")

def verify_dependencies():
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("PyTorch is not installed.")
        sys.exit(1)

    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers is not installed.")
        sys.exit(1)

if __name__ == "__main__":
    print("Verifying setup...")
    verify_python_version()
    verify_dependencies()
    print("Setup verification completed successfully.")