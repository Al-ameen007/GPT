# 🤖 GPT-2 Training & Inference Framework

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)

A modular Python framework for training GPT-2 models from scratch or fine-tuning them for text generation and classification tasks.

## 📑 Table of Contents

* [Features](#-features)
* [Prerequisites](#-prerequisites)
* [Setup & Data](#-setup--data)
* [Quick Start / Step-by-Step Guide](#-quick-start--step-by-step-guide)
* [CLI Arguments & Options](#%EF%B8%8F-cli-arguments--options)
* [Model Zoo & VRAM Requirements](#-model-zoo--vram-requirements)
* [Project Structure](#-project-structure)
* [Citation](#-citation)

---

## ✨ Features

* **Flexible Training:** Train GPT-2 models from scratch on custom text data or fine-tune pretrained weights.
* **Multiple Tasks:** Out-of-the-box support for both text generation and text classification.
* **Scalable Sizes:** Support for multiple model sizes (Small 124M, Medium 355M, Large 774M, XL 1558M).
* **Automated Pipeline:** Automatic downloading of OpenAI's pretrained weights and sample classification datasets (SMS Spam).
* **Efficient Transfer Learning:** Layer freezing capabilities to speed up fine-tuning.
* **Robust State Management:** Comprehensive checkpointing with full training state recovery.
* **CLI Ready:** Simple, argument-driven command-line interfaces for both training and inference.

---

## 🛠 Prerequisites

**Python 3.8+** is required. Install all necessary dependencies:

```bash
pip install torch tiktoken pandas numpy tqdm requests tensorflow
