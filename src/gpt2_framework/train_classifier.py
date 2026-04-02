import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader

from gpt2_framework.checkpoint import save_model
from gpt2_framework.config import (
    GPT_CONFIG_LARGE,
    GPT_CONFIG_MEDIUM,
    GPT_CONFIG_SMALL,
    GPT_CONFIG_XLARGE,
)
from gpt2_framework.data import (
    SpamDataset,
    create_balanced_dataset,
    download_and_unzip_spam_data,
    random_split,
)
from gpt2_framework.engine import calc_accuracy_loader, train_classifier_simple
from gpt2_framework.model import GPTClassifier
from gpt2_framework.weights import load_weights_into_classifier


def main():
    parser = argparse.ArgumentParser(
        description="Train a GPT classifier for text classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--url",
        type=str,
        default="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        help="URL to download dataset from",
    )
    parser.add_argument(
        "--data_dir", type=str, default="Data", help="Directory to store data files"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of data to use for training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for validation",
    )

    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "medium", "large", "xlarge"],
        help="Model size to train",
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="Number of classification classes"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained GPT-2 weights before training",
    )
    parser.add_argument(
        "--gpt2_models_dir",
        type=str,
        default="gpt2",
        help="Directory to store/load pretrained GPT-2 models",
    )
    parser.add_argument(
        "--freeze_base",
        action="store_true",
        help="Freeze all layers except the last transformer block and classification head (only with --pretrained)",
    )

    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=50, help="Evaluate every N training steps"
    )
    parser.add_argument(
        "--eval_iter",
        type=int,
        default=5,
        help="Number of batches to use for evaluation",
    )

    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for data loading"
    )

    parser.add_argument(
        "--output_dir", type=str, default="models", help="Directory to save model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt_classifier",
        help="Name for saved model files",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If None, auto-detects",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = "sms_spam_collection.zip"
    extracted_path = data_dir / "sms_spam_collection"
    data_file_path = extracted_path / "SMSSpamCollection.tsv"

    print("\nDownloading and extracting data...")
    download_and_unzip_spam_data(args.url, zip_path, extracted_path, data_file_path)

    print("Loading data...")
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

    print("Creating balanced dataset...")
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    print("Splitting data...")
    train_df, val_df, test_df = random_split(
        balanced_df, args.train_ratio, args.val_ratio
    )

    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"

    train_df.to_csv(train_csv, index=None)
    val_df.to_csv(val_csv, index=None)
    test_df.to_csv(test_csv, index=None)

    tokenizer = tiktoken.get_encoding("gpt2")

    print("\nCreating datasets...")
    train_dataset = SpamDataset(train_csv, tokenizer)
    val_dataset = SpamDataset(val_csv, tokenizer, train_dataset.max_length)
    test_dataset = SpamDataset(test_csv, tokenizer, train_dataset.max_length)

    print(f"Max sequence length: {train_dataset.max_length}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print(f"\nUsing {args.model_size.upper()} model configuration")

    configs_map = {
        "small": GPT_CONFIG_SMALL,
        "medium": GPT_CONFIG_MEDIUM,
        "large": GPT_CONFIG_LARGE,
        "xlarge": GPT_CONFIG_XLARGE,
    }

    config = deepcopy(configs_map[args.model_size])

    # IMPORTANT: Set qkv_bias based on pretrained flag
    if args.pretrained:
        config["qkv_bias"] = True  # GPT-2 pretrained weights use bias
        print("Setting qkv_bias=True for pretrained weights compatibility")
    else:
        config["qkv_bias"] = False  # Train from scratch without bias
        print("Setting qkv_bias=False for training from scratch")

    print(f"\n{'=' * 60}")
    print("Model Configuration")
    print(f"{'=' * 60}")
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print(f"{'=' * 60}\n")

    print("Initializing model...")
    model = GPTClassifier(config, num_classes=args.num_classes).to(device)

    # Load pretrained weights if requested
    if args.pretrained:
        print("\n" + "=" * 60)
        print("Loading pretrained GPT-2 weights...")
        print("=" * 60)

        try:
            from gpt2_framework.gpt_download import download_and_load_gpt2

            # Map model sizes to GPT-2 naming convention
            size_map = {
                "small": "124M",
                "medium": "355M",
                "large": "774M",
                "xlarge": "1558M",
            }
            gpt2_size = size_map[args.model_size]

            settings, params = download_and_load_gpt2(
                model_size=gpt2_size, models_dir=args.gpt2_models_dir
            )

            # Load weights into the base model (excluding the classification head)
            load_weights_into_classifier(model, params)
            print(f"✓ Loaded pretrained GPT-2 {gpt2_size} weights")

            # Optionally freeze base layers
            if args.freeze_base:
                print("\nFreezing base model parameters...")
                # Freeze all parameters first
                for param in model.parameters():
                    param.requires_grad = False

                # Unfreeze last transformer block
                for param in model.trf_blocks[-1].parameters():
                    param.requires_grad = True

                # Unfreeze classification head
                for param in model.out_head.parameters():
                    param.requires_grad = True

                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                total_params = sum(p.numel() for p in model.parameters())
                print(
                    f"✓ Trainable parameters: {trainable_params:,} / {total_params:,} "
                    f"({100 * trainable_params / total_params:.1f}%)"
                )

            print("=" * 60 + "\n")

        except ImportError as e:
            print("ERROR: Could not import required modules for pretrained weights.")
            print("Make sure 'gpt_download.py' and 'weights.py' are available.")
            print(f"Error: {e}")
            return
        except Exception as e:
            print(f"ERROR: Failed to load pretrained weights: {e}")
            return
    else:
        print("Training from scratch (random initialization)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    print(
        f"\nOptimizer: AdamW(lr={args.learning_rate}, weight_decay={args.weight_decay})"
    )

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")

    train_losses, val_losses, train_accs, val_accs, examples_seen = (
        train_classifier_simple(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            optimizer=optimizer,
            eval_freq=args.eval_freq,
            eval_iter=args.eval_iter,
            num_epochs=args.num_epochs,
        )
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60 + "\n")

    # Save final model
    final_model_path = output_dir / f"{args.model_name}.pt"
    print(f"Saving final model to {final_model_path}...")
    save_model(model, final_model_path, config)

    # Save final checkpoint
    final_checkpoint_path = output_dir / f"{args.model_name}_checkpoint.pt"
    print(f"Saving final checkpoint to {final_checkpoint_path}...")

    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.num_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "examples_seen": examples_seen,
        "config": config,
    }

    torch.save(checkpoint_data, final_checkpoint_path)

    print("\nEvaluating on all splits...")
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    if train_losses and val_losses:
        print(f"\nFinal training loss: {train_losses[-1]:.4f}")
        print(f"Final validation loss: {val_losses[-1]:.4f}")
    else:
        print("\nNo loss metrics recorded (eval_freq may be too high)")
    print(f"Total examples seen: {examples_seen:,}")

    print("\nTraining completed successfully! 🎉")


if __name__ == "__main__":
    main()
