import argparse

import tiktoken
import torch

from gpt2_framework.config import (
    GPT_CONFIG_LARGE,
    GPT_CONFIG_MEDIUM,
    GPT_CONFIG_SMALL,
    GPT_CONFIG_XLARGE,
)
from gpt2_framework.generation import generate, text_to_token, token_ids_to_text
from gpt2_framework.model import GPTClassifier, GPTModel


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with GPT models for generation or classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["generation", "classification"],
        help="Task type: generation or classification",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model file"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input text for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If None, auto-detects",
    )

    # Model configuration (if not saved in checkpoint)
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "medium", "large", "xlarge"],
        help="Model size (only needed if config not saved in checkpoint)",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=1024,
        help="Context length (only needed if config not saved in checkpoint)",
    )
    parser.add_argument(
        "--qkv_bias",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Use QKV bias (default: True for pretrained models). Use --qkv_bias false to disable.",
    )

    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="[Generation] Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="[Generation] Sampling temperature (0.0 = greedy, higher = more random)",
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="[Generation] Top-k sampling parameter"
    )

    # Classification arguments
    parser.add_argument(
        "--num_classes", type=int, default=2, help="[Classification] Number of classes"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="[Classification] Maximum sequence length for padding",
    )
    parser.add_argument(
        "--pad_token_id",
        type=int,
        default=50256,
        help="[Classification] Padding token ID",
    )

    args = parser.parse_args()

    # Device setup
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Task: {args.task}\n")

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Try to get config from checkpoint, otherwise create manually
    if "config" in checkpoint and checkpoint["config"] is not None:
        config = checkpoint["config"]
        print("✓ Config loaded from checkpoint")
    else:
        print("⚠ Config not found in checkpoint, creating from arguments...")

        # Map model sizes to configs
        configs_map = {
            "small": GPT_CONFIG_SMALL,
            "medium": GPT_CONFIG_MEDIUM,
            "large": GPT_CONFIG_LARGE,
            "xlarge": GPT_CONFIG_XLARGE,
        }

        config = configs_map[args.model_size].copy()
        config["context_length"] = args.context_length
        config["qkv_bias"] = args.qkv_bias

        print(
            f"Using {args.model_size} config with context_length={args.context_length}, qkv_bias={args.qkv_bias}"
        )

    # Load model
    if args.task == "generation":
        model = GPTModel(config).to(device)
    else:  # classification
        model = GPTClassifier(config, num_classes=args.num_classes).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("✓ Model loaded successfully!\n")

    # Get model's supported context length
    supported_context_length = model.pos_emb.weight.shape[0]

    # ==================== GENERATION MODE ====================
    if args.task == "generation":
        print("=" * 60)
        print("Text Generation")
        print("=" * 60)
        print(f"\nInput: {args.input}\n")

        # Use existing generation functions
        encoded = text_to_token(args.input, tokenizer).to(device)

        with torch.no_grad():
            token_ids = generate(
                model=model,
                idx=encoded,
                max_new_tokens=args.max_new_tokens,
                context_size=supported_context_length,
                temp=args.temperature,
                top_k=args.top_k,
                eos_id=tokenizer.encode(
                    "<|endoftext|>", allowed_special={"<|endoftext|>"}
                )[0],
            )

        generated = token_ids_to_text(token_ids, tokenizer)

        print("Generated Text:")
        print("-" * 60)
        print(generated)
        print("-" * 60)

    # ==================== CLASSIFICATION MODE ====================
    elif args.task == "classification":
        print("=" * 60)
        print("Spam Classification")
        print("=" * 60)
        print(f"\nInput: {args.input}\n")

        # Set max_length if not provided
        max_length = args.max_length if args.max_length else supported_context_length

        # Reuse logic from notebook's classify_review
        input_ids = tokenizer.encode(args.input)
        actual_length = min(len(input_ids), max_length, supported_context_length)

        input_ids = input_ids[:actual_length]
        input_ids += [args.pad_token_id] * (max_length - len(input_ids))
        input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)[:, actual_length - 1, :]

        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0, predicted_class].item()

        # Map prediction to label
        if args.num_classes == 2:
            prediction = "SPAM" if predicted_class == 1 else "NOT SPAM"
            class_labels = ["Not Spam", "Spam"]
        else:
            prediction = f"Class {predicted_class}"
            class_labels = [f"Class {i}" for i in range(args.num_classes)]

        print("Result:")
        print("-" * 60)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2%}")
        print("\nProbabilities:")
        for i, label in enumerate(class_labels):
            print(f"  {label}: {probabilities[0, i].item():.2%}")
        print("-" * 60)


if __name__ == "__main__":
    main()
