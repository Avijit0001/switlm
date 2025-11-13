"""Dataset loading and handling for SwitLM"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset as hf_load_dataset
import gc


class TextDataset(Dataset):
    """PyTorch dataset for text data"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


def load_dataset_texts(dataset_name, split_size=10000):
    """
    Load texts from a dataset
    
    Args:
        dataset_name: Name of the dataset to load
        split_size: Maximum number of samples to load
        
    Returns:
        List of text strings
    """
    print(f"📚 Loading dataset: {dataset_name}")
    
    try:
        if dataset_name == "wikitext":
            return _load_wikitext(split_size)
        elif dataset_name == "ag_news":
            return _load_ag_news(split_size)
        elif dataset_name == "imdb":
            return _load_imdb(split_size)
        elif dataset_name == "squad":
            return _load_squad(split_size)
        elif dataset_name == "tiny_stories":
            return _load_tiny_stories(split_size)
        elif dataset_name == "openwebtext":
            return _load_openwebtext(split_size)
        elif dataset_name == "c4":
            return _load_c4(split_size)
        elif dataset_name == "bookcorpus":
            return _load_bookcorpus(split_size)
        elif dataset_name == "pile":
            return _load_pile(split_size)
        else:
            print(f"❌ Unknown dataset: {dataset_name}")
            return []
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {str(e)}")
        return []
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def _load_wikitext(split_size):
    """Load WikiText dataset"""
    try:
        dataset = hf_load_dataset("wikitext", "wikitext-103-v1", split="train")
        texts = [item['text'] for item in dataset if len(item['text']) > 100][:split_size]
    except Exception as e:
        print(f"Trying wikitext-2-v1... ({str(e)})")
        dataset = hf_load_dataset("wikitext", "wikitext-2-v1", split="train")
        texts = [item['text'] for item in dataset if len(item['text']) > 100][:split_size]
    return texts


def _load_ag_news(split_size):
    """Load AG News dataset"""
    dataset = hf_load_dataset("ag_news", split=f"train[:{split_size}]")
    texts = [item["text"] for item in dataset]
    return texts


def _load_imdb(split_size):
    """Load IMDB dataset"""
    dataset = hf_load_dataset("imdb", split=f"train[:{split_size//2}]")
    texts = [item["text"] for item in dataset]
    return texts


def _load_squad(split_size):
    """Load SQuAD dataset"""
    dataset = hf_load_dataset("squad", split=f"train[:{split_size//2}]")
    texts = []
    for item in dataset:
        if "context" in item and len(item["context"]) > 100:
            texts.append(item["context"])
        if "question" in item and len(item["question"]) > 10:
            texts.append(item["question"])
    return texts


def _load_tiny_stories(split_size):
    """Load TinyStories dataset"""
    dataset = hf_load_dataset("roneneldan/TinyStories", split=f"train[:{split_size//5}]")
    texts = [item["text"] for item in dataset if len(item["text"]) > 100]
    return texts[:min(len(texts), 5000)]


def _load_openwebtext(split_size):
    """Load OpenWebText dataset"""
    try:
        dataset = hf_load_dataset("openwebtext", split=f"train[:{split_size}]")
        texts = [item['text'] for item in dataset if len(item['text']) > 100]
    except Exception:
        dataset = hf_load_dataset("openwebtext", split=f"train[:{split_size//2}]")
        texts = [item['text'] for item in dataset if len(item['text']) > 100]
    return texts


def _load_c4(split_size):
    """Load C4 dataset"""
    dataset = hf_load_dataset("c4", "en", split=f"train[:{split_size}]")
    texts = [item['text'] for item in dataset if len(item['text']) > 100]
    return texts


def _load_bookcorpus(split_size):
    """Load BookCorpus dataset"""
    dataset = hf_load_dataset("bookcorpus", split=f"train[:{split_size//2}]")
    texts = [item['text'] for item in dataset if len(item['text']) > 100]
    return texts


def _load_pile(split_size):
    """Load The Pile dataset"""
    dataset = hf_load_dataset("EleutherAI/pile", split=f"train[:{split_size//2}]")
    texts = [item['text'] for item in dataset if len(item['text']) > 100]
    return texts