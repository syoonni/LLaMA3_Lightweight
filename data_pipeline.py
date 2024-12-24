import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer, AutoTokenizer
from typing import Tuple, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TinyStoriesDatasetFactory:
    def __init__(
        self,
        tokenizer_path: str,
        max_length: int = 32,
        batch_size: int = 1,
        randomize_order: bool = True
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            padding_side="right",
            trust_remote_code=True,
            #use_flash_attention=False
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.batch_size = batch_size
        self.randomize_order = randomize_order

    def create(self, train_file: str, test_file: str = None):
        
        train_dataset = TinyStoriesDataset(
            file_path=train_file,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.randomize_order,
            num_workers=2,
            pin_memory=True
        )

        test_loader = None
        if test_file:
            test_dataset = TinyStoriesDataset(
                file_path=test_file,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )

        return train_loader, test_loader


class TinyStoriesDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stories = self._load_and_preprocess(file_path)

    def _load_and_preprocess(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    
        stories = content.split('<|endoftext|>')
        processed_stories = []
        
        for story in stories:
            story = story.strip()
            if story: 
                story = ' '.join(story.split())
                processed_stories.append(story)
        
        return processed_stories

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        
        encodings = self.tokenizer(
            story,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
    
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100

        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }