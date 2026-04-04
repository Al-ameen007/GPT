# GPT-2 Training & Inference Framework

A modular Python framework for training GPT-2 models from scratch or fine-tuning them for text generation and classification tasks.

## Features

- Train GPT-2 models from scratch on custom text data
- Fine-tune pretrained GPT-2 for text generation or classification
- Support for multiple model sizes (Small 124M, Medium 355M, Large 774M, XL 1558M)
- Automatic pretrained weight downloading from OpenAI
- Automatic spam dataset downloading for classification
- Layer freezing for efficient transfer learning
- CLI interface for training and inference
- Checkpointing with full training state recovery

## Prerequisites

**Python 3.8+** is required. Install all dependencies:

```bash
pip install torch tiktoken pandas numpy tqdm requests tensorflow
```

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `tiktoken` | GPT-2 BPE tokenizer |
| `pandas` | Data loading for classification |
| `numpy` | Weight conversion |
| `tqdm` | Progress bars |
| `requests` | Downloading pretrained weights |
| `tensorflow` | Loading OpenAI's pretrained GPT-2 checkpoints |

> **Note:** `tensorflow` is only needed when using `--pretrained` to load OpenAI's GPT-2 weights. If you only train from scratch, you can skip it.

## Project Structure

```
.
├── model.py              # GPT-2 architecture (GPTModel, GPTClassifier)
├── config.py             # Model configurations (Small/Medium/Large/XL)
├── data.py               # Dataset classes and data loading utilities
├── engine.py             # Training and evaluation loops
├── generation.py         # Text generation and classification utilities
├── checkpoint.py         # Model saving/loading
├── weights.py            # Pretrained weight loading into models
├── gpt_download.py       # Download GPT-2 weights from OpenAI
├── train.py              # CLI: train generation models
├── train_classifier.py   # CLI: train classification models
├── inference.py          # CLI: run inference (generation or classification)
├── instruction-data.json # Instruction-tuning dataset (optional)
└── README.md
```

## Setup After Cloning

After cloning, the following directories do **not** exist and are created automatically by the training scripts:

| Directory | Created By | Contents |
|-----------|-----------|----------|
| `Data/` | `train_classifier.py` | SMS Spam dataset (auto-downloaded) |
| `gpt2/` | `train.py --pretrained` or `train_classifier.py --pretrained` | Pretrained GPT-2 weights (auto-downloaded from OpenAI) |
| `models/` | `train_classifier.py` | Saved classifier models |
| `outputs/` | `train.py` | Saved generation models |

**You do not need to manually create these directories or download any data.** Everything is handled automatically when you run the training scripts.

### Getting Training Data

**For text generation (`train.py`):**
You need to provide your own text file. Any `.txt` file works:

```bash
# Example: use any text file you have
python train.py --data_path my_text.txt --model_size small --num_epochs 10
```

**For classification (`train_classifier.py`):**
The SMS Spam Collection dataset is downloaded automatically on the first run:

```bash
# Just run it - data downloads automatically
python train_classifier.py --model_size small --pretrained --freeze_base
```

### Getting Pretrained Weights

Pretrained GPT-2 weights are downloaded automatically when you use the `--pretrained` flag. The weights are saved to `gpt2/<size>/` (e.g., `gpt2/124M/`, `gpt2/355M/`).

```bash
# This auto-downloads GPT-2 Small (124M) weights (~500MB)
python train.py --data_path my_text.txt --pretrained --model_size small

# This auto-downloads GPT-2 Medium (355M) weights (~1.4GB)
python train.py --data_path my_text.txt --pretrained --model_size medium
```

You can also download weights manually using Python:

```python
from gpt_download import download_and_load_gpt2

# Downloads to gpt2/124M/
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

## Quick Start

### 1. Train a Generation Model (from scratch)

```bash
python train.py --data_path my_text.txt --model_size small --num_epochs 10 --batch_size 4
```

### 2. Train a Generation Model (fine-tune pretrained)

```bash
python train.py --data_path my_text.txt --pretrained --model_size small --context_length 1024 --num_epochs 5
```

> **Important:** When using `--pretrained`, you **must** set `--context_length 1024` to match the GPT-2 positional embeddings. Using a different value will cause a shape mismatch error.

### 3. Train a Spam Classifier (with pretrained weights + frozen layers)

```bash
python train_classifier.py --pretrained --freeze_base --model_size small --num_epochs 5
```

### 4. Run Inference

**Text generation:**
```bash
python inference.py --task generation --model_path outputs/gpt_model_final.pt --input "Once upon a time"
```

**Spam classification:**
```bash
python inference.py --task classification --model_path models/gpt_classifier.pt --input "You won a prize!"
```

## Training Options

### Generation Training (`train.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to training text file | **Required** |
| `--model_size` | Model size (small/medium/large/xlarge) | `small` |
| `--pretrained` | Load pretrained GPT-2 weights | `False` |
| `--gpt2_models_dir` | Directory for pretrained weights | `gpt2` |
| `--num_epochs` | Number of training epochs | `10` |
| `--batch_size` | Batch size | `4` |
| `--learning_rate` | Learning rate | `0.0004` |
| `--weight_decay` | Weight decay | `0.1` |
| `--context_length` | Maximum sequence length (**must be 1024 with `--pretrained`**) | `256` |
| `--stride` | Sliding window stride | `128` |
| `--eval_freq` | Evaluate every N steps | `5` |
| `--eval_iter` | Batches per evaluation | `5` |
| `--start_context` | Text for sample generation during training | `Once upon a time` |
| `--output_dir` | Output directory | `outputs` |
| `--model_name` | Saved model name prefix | `gpt_model` |
| `--device` | Device (cuda/cpu, auto-detected) | `None` |
| `--seed` | Random seed | `123` |

### Classification Training (`train_classifier.py`)

| Argument | Description | Default |
|----------|-------------|---------|
| `--url` | Dataset download URL | SMS Spam Collection |
| `--data_dir` | Directory for data files | `Data` |
| `--model_size` | Model size (small/medium/large/xlarge) | `small` |
| `--pretrained` | Load pretrained GPT-2 weights | `False` |
| `--freeze_base` | Freeze all except last block + head (use with `--pretrained`) | `False` |
| `--gpt2_models_dir` | Directory for pretrained weights | `gpt2` |
| `--num_classes` | Number of classes | `2` |
| `--num_epochs` | Number of training epochs | `5` |
| `--batch_size` | Batch size | `4` |
| `--learning_rate` | Learning rate | `1e-5` |
| `--weight_decay` | Weight decay | `0.1` |
| `--eval_freq` | Evaluate every N steps | `50` |
| `--eval_iter` | Batches per evaluation | `5` |
| `--train_ratio` | Training data ratio | `0.7` |
| `--val_ratio` | Validation data ratio | `0.1` |
| `--output_dir` | Output directory | `models` |
| `--model_name` | Saved model name prefix | `gpt_classifier` |
| `--device` | Device (cuda/cpu, auto-detected) | `None` |
| `--seed` | Random seed | `123` |

## Inference Options

### Generation

```bash
python inference.py \
    --task generation \
    --model_path outputs/gpt_model_final.pt \
    --input "Your prompt" \
    --max_new_tokens 100 \
    --temperature 0.7 \
    --top_k 50
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--max_new_tokens` | Tokens to generate | `100` |
| `--temperature` | Sampling temperature (0=greedy) | `0.7` |
| `--top_k` | Top-k sampling | `50` |

### Classification

```bash
python inference.py \
    --task classification \
    --model_path models/gpt_classifier.pt \
    --input "Your text"
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--num_classes` | Number of classes | `2` |
| `--max_length` | Max sequence length for padding | Model's context length |
| `--pad_token_id` | Padding token ID | `50256` |

## Model Sizes

| Size | Parameters | Layers | Embedding Dim | Heads | Pretrained Weight Download |
|------|-----------|--------|---------------|-------|---------------------------|
| Small | 124M | 12 | 768 | 12 | ~500 MB |
| Medium | 355M | 24 | 1024 | 16 | ~1.4 GB |
| Large | 774M | 36 | 1280 | 20 | ~3 GB |
| XL | 1558M | 48 | 1600 | 25 | ~6 GB |

## Reproducing Results Step by Step

### From a fresh clone (no Data/, gpt2/, models/ directories):

**1. Install dependencies:**
```bash
pip install torch tiktoken pandas numpy tqdm requests tensorflow
```

**2. Train a spam classifier with pretrained GPT-2 (recommended first run):**
```bash
python train_classifier.py --pretrained --freeze_base --num_epochs 5
```
This will:
- Auto-download the SMS Spam dataset into `Data/`
- Auto-download GPT-2 Small weights into `gpt2/124M/`
- Train a classifier and save it to `models/gpt_classifier.pt`

**3. Test the classifier:**
```bash
python inference.py --task classification --model_path models/gpt_classifier.pt --input "Congratulations! You won $1000!"
python inference.py --task classification --model_path models/gpt_classifier.pt --input "Hey, are we still meeting for lunch?"
```

**4. Train a generation model (provide your own text file):**
```bash
python train.py --data_path your_text.txt --pretrained --model_size small --context_length 1024 --batch_size 1 --num_epochs 10
```
> Use `--batch_size 1` if your GPU has 6 GB or less VRAM.

**5. Generate text:**
```bash
python inference.py --task generation --model_path outputs/gpt_model_final.pt --input "Once upon a time"
```

## Tips

**Training:**
- Start with `small` model and `--pretrained --freeze_base` for fast results
- When using `--pretrained`, `--context_length` **must be 1024** (GPT-2's native context length)
- Use `--context_length 256` for faster training only when training from scratch (without `--pretrained`)
- If you have limited VRAM (e.g. 6 GB), reduce `--batch_size 1` when using pretrained models with context_length=1024
- Increase `--batch_size` if you have more GPU memory
- Use lower learning rates for classification (1e-5 to 5e-5)

**Generation:**
- `temperature=0.0` = deterministic (greedy decoding)
- `temperature=0.7` = balanced creativity
- `temperature=1.5` = very creative/random
- Lower `top_k` for more focused output

**Classification:**
- `--freeze_base` trains much faster (only ~5% of parameters are updated)
- Fewer epochs needed (3-5 is usually enough)

## Device Support & VRAM Requirements

CUDA is auto-detected. Force a specific device:
```bash
python train.py --device cpu --data_path data.txt
python train.py --device cuda --data_path data.txt
```

| Model Size | context_length | batch_size | Approx. VRAM |
|------------|---------------|------------|--------------|
| Small | 1024 | 1 | ~4 GB |
| Small | 1024 | 4 | ~8 GB |
| Medium | 1024 | 1 | ~8 GB |
| Medium | 1024 | 4 | ~16 GB |

If you run out of VRAM, reduce `--batch_size` first. If still OOM with batch_size=1, use `--device cpu`.

## Checkpointing

Both training scripts save:
- **Model weights** (`.pt`) - for inference
- **Full checkpoint** (`_checkpoint.pt`) - for resuming training (includes optimizer state, loss history, config)

## License

MIT License

## Citation

Based on the GPT-2 architecture from:
```
Radford et al., "Language Models are Unsupervised Multitask Learners", 2019
```
