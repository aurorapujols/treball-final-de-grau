import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Sets random seeds for reproducibility across:
    - Python
    - NumPy
    - PyTorch (CPU + CUDA)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed] Using seed: {seed}")
