import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken
from typing import List, Tuple, Optional
import pandas as pd
import urllib.request
import zipfile
import os
from pathlib import Path


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []
        
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizer = None
) -> DataLoader:
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    
    dataset = GPTDatasetV1(
        txt=txt,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


def create_dataloaders(
    train_data: str,
    val_data: str,
    batch_size: int = 4,
    max_length: int = 256,
    train_stride: int = 128,
    val_stride: Optional[int] = None,
    num_workers: int = 0,
    tokenizer = None
) -> Tuple[DataLoader, DataLoader]:
    if val_stride is None:
        val_stride = max_length
    
    train_loader = create_dataloader_v1(
        txt=train_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=train_stride,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        tokenizer=tokenizer
    )
    
    val_loader = create_dataloader_v1(
        txt=val_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=val_stride,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        tokenizer=tokenizer
    )
    
    return train_loader, val_loader


def split_text_data(text: str, train_ratio: float = 0.9) -> Tuple[str, str]:
    split_idx = int(train_ratio * len(text))
    train_data = text[:split_idx]
    val_data = text[split_idx:]
    
    return train_data, val_data


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def create_balanced_dataset(df):
    num_samp = df[df['Label'] == 'spam'].shape[0]
    ham_subset = df[df['Label'] == 'ham'].sample(num_samp, random_state=123)
    return pd.concat([ham_subset, df[df['Label'] == 'spam']])


def random_split(df, train_frac, val_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    return train_df, val_df, test_df


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.encoded_text = [
            tokenizer.encode(text) for text in self.data['Text']
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_text = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_text
            ]

        self.encoded_text = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for
            encoded_text in self.encoded_text
        ]

    def __getitem__(self, index):
        encoded = self.encoded_text[index]
        label = self.data.iloc[index]['Label']
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_text:
            if max_length < len(encoded_text):
                max_length = len(encoded_text)
        return max_length