import torch
from torch.utils.data import Dataset
from typing import Optional, List, Union, Any

class HFDatasetAdapter(Dataset):
    def __init__(
        self,
        hf_dataset: Any,
        feature_col: str = 'image',
        target_col: Optional[str] = 'label',
        transform: Optional[Any] = None
    ):
        self.dataset = hf_dataset
        self.feature_col = feature_col
        self.target_col = target_col
        self.transform = transform
        
        if self.feature_col not in self.dataset.column_names:
            raise ValueError(f"Feature column '{self.feature_col}' not found in dataset. Available: {self.dataset.column_names}")
        
        if self.target_col and self.target_col not in self.dataset.column_names:
            print(f"⚠️ Warning: Target column '{self.target_col}' not found. It will be ignored.")
            self.target_col = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        feature = item[self.feature_col]
        target = item[self.target_col] if self.target_col else None
        
        if self.transform:
            feature = self.transform(feature)
            
        if self.target_col:
            return feature, target
        return feature

    @staticmethod
    def from_hub(path: str, split: str = 'train', **kwargs):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("'datasets' library is required. Install it with `pip install datasets`.")
            
        dataset = load_dataset(path, split=split, **kwargs)
        return dataset
