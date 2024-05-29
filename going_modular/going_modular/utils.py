"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
import os
def save_model(model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               saved_dir: str,
               model_name: str,
               epoch: int,
               train_loss: float,
               test_acc: float):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(saved_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'acc': test_acc,
                }, model_save_path)

    # torch.save(obj=model.state_dict(),
    #          f=model_save_path)

