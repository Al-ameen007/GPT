"""
checkpoint.py - Model Saving & Loading Utilities

This module provides functions for saving and loading model checkpoints during training,
as well as saving and loading trained model weights.
"""

import torch
import os
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, 
                   tokens_seen, config, filepath):
    """
    Save a complete training checkpoint including model state, optimizer state, 
    training progress, and configuration.
    
    Args:
        model: The PyTorch model to save
        optimizer: The optimizer with its state
        epoch: Current epoch number
        train_losses: List of training losses
        val_losses: List of validation losses
        tokens_seen: List tracking tokens seen during training
        config: Model configuration dictionary
        filepath: Path where to save the checkpoint (e.g., 'checkpoints/model_epoch_10.pt')
    
    Example:
        >>> save_checkpoint(
        ...     model=gpt_model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     train_losses=train_loss_list,
        ...     val_losses=val_loss_list,
        ...     tokens_seen=tokens_list,
        ...     config=GPT_CONFIG_SMALL,
        ...     filepath='checkpoints/gpt_epoch_10.pt'
        ... )
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'tokens_seen': tokens_seen,
        'config': config
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """
    Load a complete training checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        model: The model to load weights into
        optimizer: (Optional) The optimizer to load state into
        device: Device to map the model to ('cpu' or 'cuda')
    
    Returns:
        Dictionary containing:
            - epoch: The epoch number when checkpoint was saved
            - train_losses: Training loss history
            - val_losses: Validation loss history
            - tokens_seen: Tokens seen history
            - config: Model configuration
    
    Example:
        >>> model = GPT2(GPT_CONFIG_SMALL)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004)
        >>> checkpoint_data = load_checkpoint(
        ...     filepath='checkpoints/gpt_epoch_10.pt',
        ...     model=model,
        ...     optimizer=optimizer,
        ...     device='cuda'
        ... )
        >>> start_epoch = checkpoint_data['epoch'] + 1
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    
    return {
        'epoch': checkpoint['epoch'],
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'tokens_seen': checkpoint['tokens_seen'],
        'config': checkpoint['config']
    }


def save_model(model, filepath, config=None):
    """
    Save only the model weights (lighter than full checkpoint).
    Useful for saving final trained models or for inference.
    
    Args:
        model: The PyTorch model to save
        filepath: Path where to save the model (e.g., 'models/gpt_final.pt')
        config: (Optional) Model configuration dictionary to save alongside weights
    
    Example:
        >>> save_model(
        ...     model=gpt_model,
        ...     filepath='models/gpt_trained.pt',
        ...     config=GPT_CONFIG_SMALL
        ... )
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict()
    }
    
    if config is not None:
        save_dict['config'] = config
    
    torch.save(save_dict, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, model, device='cpu'):
    """
    Load only model weights (no optimizer state or training history).
    
    Args:
        filepath: Path to the saved model file
        model: The model to load weights into
        device: Device to map the model to ('cpu' or 'cuda')
    
    Returns:
        config: Model configuration if it was saved, otherwise None
    
    Example:
        >>> model = GPT2(GPT_CONFIG_SMALL)
        >>> config = load_model(
        ...     filepath='models/gpt_trained.pt',
        ...     model=model,
        ...     device='cuda'
        ... )
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded from {filepath}")
    
    config = checkpoint.get('config', None)
    return config


def save_best_model(model, optimizer, epoch, train_loss, val_loss, 
                    best_val_loss, filepath='checkpoints/best_model.pt', 
                    config=None):
    """
    Save model only if validation loss improved (best model tracking).
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        epoch: Current epoch
        train_loss: Current training loss
        val_loss: Current validation loss
        best_val_loss: Best validation loss seen so far
        filepath: Path where to save the best model
        config: Model configuration
    
    Returns:
        Updated best_val_loss (either the new val_loss if improved, or the previous best)
    
    Example:
        >>> best_val_loss = float('inf')
        >>> for epoch in range(num_epochs):
        ...     # ... training code ...
        ...     best_val_loss = save_best_model(
        ...         model=model,
        ...         optimizer=optimizer,
        ...         epoch=epoch,
        ...         train_loss=current_train_loss,
        ...         val_loss=current_val_loss,
        ...         best_val_loss=best_val_loss,
        ...         config=GPT_CONFIG_SMALL
        ...     )
    """
    if val_loss < best_val_loss:
        previous_best = best_val_loss
        best_val_loss = val_loss

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }

        torch.save(checkpoint, filepath)
        print(f"✓ New best model saved! Val loss: {val_loss:.4f} (previous: {previous_best:.4f})")
    
    return best_val_loss


def list_checkpoints(directory='checkpoints'):
    """
    List all checkpoint files in a directory.
    
    Args:
        directory: Directory to search for checkpoints
    
    Returns:
        List of checkpoint file paths
    
    Example:
        >>> checkpoints = list_checkpoints('checkpoints')
        >>> for ckpt in checkpoints:
        ...     print(ckpt)
    """
    checkpoint_dir = Path(directory)
    
    if not checkpoint_dir.exists():
        print(f"Directory {directory} does not exist")
        return []
    
    checkpoints = sorted(checkpoint_dir.glob('*.pt'))
    
    if not checkpoints:
        print(f"No checkpoints found in {directory}")
        return []
    
    print(f"Found {len(checkpoints)} checkpoint(s) in {directory}:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.name}")
    
    return [str(ckpt) for ckpt in checkpoints]