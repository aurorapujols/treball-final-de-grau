import torch
import os

def save_checkpoint(model, path):
    """
    Saves a model checkpoint to the given path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Checkpoint] Saved model to: {path}")


def load_checkpoint(model, path, device="cpu"):
    """
    Loads a model checkpoint into the given model.
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    print(f"[Checkpoint] Loaded model from: {path}")
    return model
