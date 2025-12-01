import torch
import numpy as np
import threading
import queue
import time
from typing import Any, Callable, Dict, List, Optional, Union, Iterable, Tuple
from .samplers import Sampler, SequentialSampler, RandomSampler, BatchSampler

def default_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, (tuple, list)):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    
    raise TypeError(f'default_collate: batch type not supported {elem_type}')

class FlashDataset:
    def __init__(self, data_source: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], memory_optimized: bool = True, labels: Optional[List[str]] = None):
        self.memory_optimized = memory_optimized
        self.labels = labels
        
        if isinstance(data_source, tuple):
            self.features, self.targets = data_source
        else:
            self.features = data_source
            self.targets = None
            
        if self.memory_optimized:
            if isinstance(self.features, torch.Tensor) and self.features.dtype == torch.float32:
                self.features = self.features.half()
            
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        feature = self.features[idx]
        
        if self.memory_optimized and isinstance(feature, torch.Tensor) and feature.dtype == torch.float16:
            feature = feature.float()
            
        if self.targets is not None:
            target = self.targets[idx]
            return feature, target
        return feature

    def save(self, path: str):
        torch.save({
            'features': self.features,
            'targets': self.targets,
            'labels': self.labels,
            'memory_optimized': self.memory_optimized
        }, path)
        
    @staticmethod
    def load(path: str) -> 'FlashDataset':
        data = torch.load(path)
        
        if isinstance(data, dict):
            dataset = FlashDataset(
                data_source=(data['features'], data['targets']) if data['targets'] is not None else data['features'],
                memory_optimized=data.get('memory_optimized', True),
                labels=data.get('labels')
            )
        elif isinstance(data, (tuple, list)):
            if len(data) == 2:
                dataset = FlashDataset(data_source=data, memory_optimized=True)
            else:
                dataset = FlashDataset(data_source=data[0], memory_optimized=True)
        elif isinstance(data, torch.Tensor):
            dataset = FlashDataset(data_source=data, memory_optimized=True)
        else:
            raise TypeError(f"File format not supported: {type(data)}")
            
        return dataset

class FlashDataLoader:
    def __init__(
        self,
        dataset: Union[FlashDataset, 'ChunkedFlashDataset'],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: int = 2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        chunk_batch_size: Optional[int] = None,
        batch_transform: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.device = device
        self.batch_transform = batch_transform
        self.collate_fn = collate_fn if collate_fn is not None else default_collate
        self.shuffle = shuffle
        
        try:
            from .chunked_dataset import ChunkedFlashDataset
            self.is_chunked = isinstance(dataset, ChunkedFlashDataset)
        except ImportError:
            self.is_chunked = False
        
        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and drop_last')
            self.batch_size = 1
            self.drop_last = False
            self.sampler = None
            self.batch_sampler = batch_sampler
        else:
            if batch_size is None:
                self.batch_size = 1
            else:
                self.batch_size = batch_size
                
            self.drop_last = drop_last
            
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            self.sampler = sampler
            
            self.batch_sampler = BatchSampler(self.sampler, self.batch_size, self.drop_last)

        self.chunk_batch_size = chunk_batch_size or self.batch_size
        
        self.batch_queue = queue.Queue(maxsize=self.prefetch_factor * (num_workers if num_workers > 0 else 1))
        self.stop_event = threading.Event()
        self.prefetch_thread = None
        
        self.current_chunk_samples = []
        self.current_chunk_targets = []
        self.current_chunk_idx = -1
    
    def _chunked_iter(self):
        print("Using optimized chunked mode (Complex samplers ignored in pure chunked mode)")
        
        current_idx = 0
        total_len = len(self.dataset)
        
        while current_idx < total_len:
            remaining = total_len - current_idx
            if remaining < self.batch_size:
                if self.drop_last:
                    break
                real_batch_size = remaining
            else:
                real_batch_size = self.batch_size
                
            batch_indices = range(current_idx, current_idx + real_batch_size)
            current_idx += real_batch_size
            
            batch_samples = []
            for idx in batch_indices:
                batch_samples.append(self.dataset[idx])
            
            batch = self.collate_fn(batch_samples)
            
            yield self._process_batch(batch)

    def _process_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            features, targets = batch
        else:
            features = batch
            targets = None
            
        if self.pin_memory and self.device == 'cuda':
            if isinstance(features, torch.Tensor):
                features = features.pin_memory()
            if isinstance(targets, torch.Tensor):
                targets = targets.pin_memory()
        
        if isinstance(features, torch.Tensor):
            features = features.to(self.device, non_blocking=True)
        if isinstance(targets, torch.Tensor):
            targets = targets.to(self.device, non_blocking=True)
            
        if self.batch_transform:
            if targets is not None:
                try:
                    res = self.batch_transform(features, targets)
                    if isinstance(res, tuple) and len(res) == 2:
                        features, targets = res
                    else:
                        features = res
                except TypeError:
                    features = self.batch_transform(features)
            else:
                features = self.batch_transform(features)
                
        if targets is not None:
            return features, targets
        return features

    def __iter__(self):
        if self.is_chunked:
            return self._chunked_iter()
        elif isinstance(self.dataset, FlashDataset) and self.num_workers == 0 and self.collate_fn == default_collate:
            return self._vectorized_iter()
        elif self.num_workers > 0:
            return self._multi_worker_iter()
        else:
            return self._single_thread_iter()

    def _vectorized_iter(self):
        for batch_indices in self.batch_sampler:
            batch = self.dataset[batch_indices]
            yield self._process_batch(batch)

    def _single_thread_iter(self):
        for batch_indices in self.batch_sampler:
            batch_samples = [self.dataset[i] for i in batch_indices]
            batch = self.collate_fn(batch_samples)
            yield self._process_batch(batch)
    
    def _multi_worker_iter(self):
        self.stop_event.clear()
        self.prefetch_thread = threading.Thread(target=self._prefetch_batches, daemon=True)
        self.prefetch_thread.start()
        
        while True:
            batch = self.batch_queue.get()
            if batch is None:
                break
            yield batch
        
        self.prefetch_thread.join(timeout=1.0)
    
    def _prefetch_batches(self):
        try:
            for batch_indices in self.batch_sampler:
                if self.stop_event.is_set():
                    break
                
                batch_samples = [self.dataset[i] for i in batch_indices]
                
                batch = self.collate_fn(batch_samples)
                
                processed_batch = self._process_batch(batch)
                
                self.batch_queue.put(processed_batch)
        except Exception as e:
            print(f"Error in prefetch worker: {e}")
        finally:
            self.batch_queue.put(None)
    
    def __len__(self):
        return len(self.batch_sampler)
    
    def __del__(self):
        self.stop_event.set()
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=0.5)

def get_dataloaders(
    chunks_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    augmentations: Optional[Dict[str, Callable]] = None,
    memory_optimized: bool = True
) -> Dict[str, FlashDataLoader]:
    from .chunked_dataset import ChunkedFlashDataset
    
    loaders = {}
    splits = ['train', 'val', 'test']
    
    augmentations = augmentations or {}
    
    for split in splits:
        try:
            dataset = ChunkedFlashDataset(
                chunks_dir=chunks_dir,
                split=split,
                memory_optimized=memory_optimized,
                verbose=True if split == 'train' else False
            )
            
            if len(dataset) == 0:
                continue
                
            transform = augmentations.get(split)
            
            loader = FlashDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                device=device,
                batch_transform=transform
            )
            
            loaders[split] = loader
            print(f"✅ Loader '{split}' created: {len(dataset)} samples, {len(loader)} batches")
            
        except Exception as e:
            print(f"⚠️ Could not create loader for '{split}': {e}")
            
    return loaders