import torch
import numpy as np
import os
import json
import threading
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .core import FlashDataset
from .memory import MemoryOptimizer

class ChunkedFlashDataset(FlashDataset):
    def __init__(
        self,
        chunks_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        memory_optimized: bool = True,
        chunk_size: int = 10000,
        preload_next_chunk: bool = True,
        verbose: bool = False,
        split: str = 'train'
    ):
        self.chunks_dir = chunks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.memory_optimized = memory_optimized
        self.chunk_size = chunk_size
        self.preload_next_chunk = preload_next_chunk
        self.verbose = verbose
        self.split = split
        
        self._load_metadata()
        
        self.memory_optimizer = MemoryOptimizer() if memory_optimized else None
        
        self.current_chunk_idx = -1
        self.current_chunk = None
        self.chunk_offset = 0
        
        self.preload_queue = queue.Queue(maxsize=1) if preload_next_chunk else None
        self.preload_thread = None
        self.stop_preloading = threading.Event()
        
        if self.num_chunks > 0:
            first_chunk_idx = 0
            self._load_chunk(first_chunk_idx)
            
            if self.preload_next_chunk:
                self._start_preloading()
        else:
            if self.verbose:
                print(f"⚠️ Warning: No chunks available for split '{split}'")
    
    def _load_metadata(self):
        metadata_path = os.path.join(self.chunks_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.total_samples = metadata['total_samples']
        self.feature_shape = metadata['feature_shape']
        self.target_shape = metadata['target_shape']
        self.feature_dtype = metadata['feature_dtype']
        self.target_dtype = metadata['target_dtype']
        self.labels = metadata.get('labels', None)
        
        all_chunk_indices = metadata.get('chunk_indices', [])
        splits_mapping = metadata.get('splits', {})
        
        if self.split == 'all' or not splits_mapping:
            self.active_chunk_ids = list(range(len(all_chunk_indices)))
        else:
            if self.split not in splits_mapping:
                raise ValueError(f"Split '{self.split}' not found in metadata. Available: {list(splits_mapping.keys())}")
            self.active_chunk_ids = splits_mapping[self.split]
            
        self.num_chunks = len(self.active_chunk_ids)
        
        self.split_total_samples = 0
        self.chunk_indices = []
        
        current_offset = 0
        for chunk_id in self.active_chunk_ids:
            global_start, global_end = all_chunk_indices[chunk_id]
            chunk_len = global_end - global_start
            
            self.chunk_indices.append((current_offset, current_offset + chunk_len))
            current_offset += chunk_len
            self.split_total_samples += chunk_len
            
        if self.verbose:
            print(f"📂 Dataset loaded (Split: {self.split})")
            print(f"   Chunks: {self.num_chunks}")
            print(f"   Samples: {self.split_total_samples}")

    def _load_chunk(self, chunk_idx: int):
        if chunk_idx < 0 or chunk_idx >= self.num_chunks:
            raise ValueError(f"Invalid chunk index: {chunk_idx} (Total in split: {self.num_chunks})")
        
        real_chunk_id = self.active_chunk_ids[chunk_idx]
        
        if self.verbose:
            print(f"💾 Loading chunk {chunk_idx} (Real ID: {real_chunk_id})...")
        
        chunk_path = os.path.join(self.chunks_dir, f'chunk_{real_chunk_id}.pt')
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Chunk not found: {chunk_path}")
        
        chunk_data = torch.load(chunk_path)
        self.current_chunk = {
            'features': chunk_data['features'],
            'targets': chunk_data['targets']
        }
        self.current_chunk_idx = chunk_idx
        self.chunk_offset = self.chunk_indices[chunk_idx][0]
        
        if self.memory_optimized and self.memory_optimizer:
            if self.verbose:
                print("  Optimizing chunk memory...")
            self.current_chunk['features'] = self.memory_optimizer.optimize_tensor(self.current_chunk['features'])
            self.current_chunk['targets'] = self.memory_optimizer.optimize_tensor(self.current_chunk['targets'])
        
        if self.verbose:
            print(f"✅ Chunk {chunk_idx} loaded with {len(self.current_chunk['features'])} samples")
    
    def _start_preloading(self):
        if self.preload_thread and self.preload_thread.is_alive():
            self.stop_preloading.set()
            self.preload_thread.join(timeout=1.0)
        
        self.stop_preloading.clear()
        self.preload_thread = threading.Thread(target=self._preload_chunks, daemon=True)
        self.preload_thread.start()
        if self.verbose:
            print("🔄 Starting background chunk preloading...")
    
    def _preload_chunks(self):
        next_chunk_idx = self.current_chunk_idx + 1
        
        while not self.stop_preloading.is_set() and next_chunk_idx < self.num_chunks:
            try:
                real_chunk_id = self.active_chunk_ids[next_chunk_idx]
                if self.verbose:
                    print(f"🔮 Preloading chunk {next_chunk_idx} (Real ID: {real_chunk_id}) in background...")
                chunk_path = os.path.join(self.chunks_dir, f'chunk_{real_chunk_id}.pt')
                chunk_data = torch.load(chunk_path)
                
                if self.memory_optimized and self.memory_optimizer:
                    chunk_data['features'] = self.memory_optimizer.optimize_tensor(chunk_data['features'])
                    chunk_data['targets'] = self.memory_optimizer.optimize_tensor(chunk_data['targets'])
                
                self.preload_queue.put((next_chunk_idx, chunk_data))
                next_chunk_idx += 1
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error preloading chunk {next_chunk_idx}: {e}")
                break
    
    def _get_item_from_chunk(self, idx: int) -> Tuple[Any, Any]:
        if idx < self.chunk_offset or idx >= self.chunk_offset + len(self.current_chunk['features']):
            chunk_idx = next(i for i, (start, end) in enumerate(self.chunk_indices) if start <= idx < end)
            
            if self.preload_next_chunk and self.preload_queue and not self.preload_queue.empty():
                precached_chunk_idx, precached_data = self.preload_queue.get()
                if precached_chunk_idx == chunk_idx:
                    self.current_chunk = precached_data
                    self.current_chunk_idx = chunk_idx
                    self.chunk_offset = self.chunk_indices[chunk_idx][0]
                    if self.verbose:
                        print(f"⚡ Using preloaded chunk {chunk_idx}")
                else:
                    self._load_chunk(chunk_idx)
            else:
                self._load_chunk(chunk_idx)
        
        local_idx = idx - self.chunk_offset
        
        feature = self.current_chunk['features'][local_idx]
        target = self.current_chunk['targets'][local_idx]
        
        return feature, target
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if idx < 0 or idx >= self.split_total_samples:
            raise IndexError(f"Index {idx} out of range (0-{self.split_total_samples-1})")
        
        feature, target = self._get_item_from_chunk(idx)
        
        if self.transform:
            feature = self.transform(feature)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return feature, target
    
    def __len__(self):
        return self.split_total_samples
    
    def __del__(self):
        if self.preload_next_chunk:
            self.stop_preloading.set()
            if self.preload_thread and self.preload_thread.is_alive():
                self.preload_thread.join(timeout=1.0)