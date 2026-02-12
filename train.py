import argparse
import torch
from pathlib import Path
import tiktoken
from copy import deepcopy

from model import GPTModel
from config import (
    GPT_CONFIG_SMALL, GPT_CONFIG_MEDIUM, GPT_CONFIG_LARGE, GPT_CONFIG_XLARGE
)
from data import create_dataloaders, split_text_data
from engine import train_model_simple
from checkpoint import save_checkpoint, save_model


def main():
    parser = argparse.ArgumentParser(
        description='Train a GPT model for text generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to text file for training'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.9,
        help='Ratio of data to use for training (rest is validation)'
    )
    
    parser.add_argument(
        '--model_size',
        type=str,
        default='small',
        choices=['small', 'medium', 'large', 'xlarge'],
        help='Model size to train'
    )
    parser.add_argument(
        '--context_length',
        type=int,
        default=1024,
        help='Maximum sequence length (context window)'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Load pretrained GPT-2 weights before training'
    )
    parser.add_argument(
        '--gpt2_models_dir',
        type=str,
        default='gpt2',
        help='Directory to store/load pretrained GPT-2 models'
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0004,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.1,
        help='Weight decay for optimizer'
    )
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=5,
        help='Evaluate every N training steps'
    )
    parser.add_argument(
        '--eval_iter',
        type=int,
        default=5,
        help='Number of batches to use for evaluation'
    )
    
    parser.add_argument(
        '--stride',
        type=int,
        default=128,
        help='Stride for sliding window (lower = more overlap)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of workers for data loading'
    )
    
    parser.add_argument(
        '--start_context',
        type=str,
        default='Once upon a time',
        help='Starting text for sample generation during training'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt_model',
        help='Name for saved model files'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). If None, auto-detects'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print(f"\nLoading data from {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
    
    print(f"Total characters: {len(text_data):,}")
    
    train_data, val_data = split_text_data(text_data, args.train_ratio)
    print(f"Train characters: {len(train_data):,}")
    print(f"Validation characters: {len(val_data):,}")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        max_length=args.context_length,
        train_stride=args.stride,
        num_workers=args.num_workers,
        tokenizer=tokenizer
    )
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    print(f"\nUsing {args.model_size.upper()} model configuration")
    
    configs_map = {
        'small': GPT_CONFIG_SMALL,
        'medium': GPT_CONFIG_MEDIUM,
        'large': GPT_CONFIG_LARGE,
        'xlarge': GPT_CONFIG_XLARGE
    }
    
    config = deepcopy(configs_map[args.model_size])
    config['context_length'] = args.context_length
    
    # IMPORTANT: Set qkv_bias based on pretrained flag
    if args.pretrained:
        config['qkv_bias'] = True  # GPT-2 pretrained weights use bias
        print("Setting qkv_bias=True for pretrained weights compatibility")
    else:
        config['qkv_bias'] = False  # Train from scratch without bias
        print("Setting qkv_bias=False for training from scratch")
    
    print(f"\n{'='*60}")
    print(f"Model Configuration")
    print(f"{'='*60}")
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print(f"{'='*60}\n")
    
    print("Initializing model...")
    model = GPTModel(config).to(device)
    
    # Load pretrained weights if requested
    if args.pretrained:
        print("\n" + "="*60)
        print("Loading pretrained GPT-2 weights...")
        print("="*60)
        
        try:
            from gpt_download import download_and_load_gpt2
            from weights import load_weights_into_gpt
            
            # Map model sizes to GPT-2 naming convention
            size_map = {
                'small': '124M',
                'medium': '355M',
                'large': '774M',
                'xlarge': '1558M'
            }
            gpt2_size = size_map[args.model_size]
            
            settings, params = download_and_load_gpt2(
                model_size=gpt2_size,
                models_dir=args.gpt2_models_dir
            )
            
            load_weights_into_gpt(model, params)
            print(f"✓ Loaded pretrained GPT-2 {gpt2_size} weights")
            print("="*60 + "\n")
            
        except ImportError as e:
            print(f"ERROR: Could not import required modules for pretrained weights.")
            print(f"Make sure 'gpt_download.py' and 'weights.py' are available.")
            print(f"Error: {e}")
            return
        except Exception as e:
            print(f"ERROR: Failed to load pretrained weights: {e}")
            return
    else:
        print("Training from scratch (random initialization)")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    print(f"\nOptimizer: AdamW(lr={args.learning_rate}, weight_decay={args.weight_decay})")
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context=args.start_context,
        tokenizer=tokenizer
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60 + "\n")

    final_model_path = output_dir / f"{args.model_name}_final.pt"
    print(f"Saving final model to {final_model_path}...")
    save_model(model, final_model_path, config)

    final_checkpoint_path = output_dir / f"{args.model_name}_final_checkpoint.pt"
    print(f"Saving final checkpoint to {final_checkpoint_path}...")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.num_epochs,
        train_losses=train_losses,
        val_losses=val_losses,
        tokens_seen=tokens_seen,
        config=config,
        filepath=final_checkpoint_path
    )

    if train_losses and val_losses and tokens_seen:
        print(f"\nFinal training loss: {train_losses[-1]:.4f}")
        print(f"Final validation loss: {val_losses[-1]:.4f}")
        print(f"Total tokens seen: {tokens_seen[-1]:,}")
    else:
        print("\nNo evaluation metrics recorded (eval_freq may be too high)")
    
    print("\nTraining completed successfully! 🎉")


if __name__ == "__main__":
    main()