import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import DataLoader
import os
import gc
import json
import psutil
import zarr
from .core import FlashDataset
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import traceback

class DatasetConverter:
    
    @staticmethod
    def from_pytorch_dataset(
        pytorch_dataset: torch.utils.data.Dataset,
        memory_optimized: bool = True,
        output_path: str = None
    ) -> FlashDataset:
        print("Converting PyTorch dataset to FlashDataset format (full RAM load)...")
        
        features = []
        targets = []
        
        for i in range(len(pytorch_dataset)):
            item = pytorch_dataset[i]
            if isinstance(item, tuple) and len(item) == 2:
                feature, target = item
                features.append(feature)
                targets.append(target)
            else:
                features.append(item)
        
        features_tensor = torch.stack([torch.tensor(f) if not isinstance(f, torch.Tensor) else f for f in features])
        
        if targets:
            targets_tensor = torch.tensor(targets) if not isinstance(targets[0], torch.Tensor) else torch.stack(targets)
            data = (features_tensor, targets_tensor)
        else:
            data = features_tensor
        
        flash_dataset = FlashDataset(
            data_source=data,
            memory_optimized=memory_optimized
        )
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            torch.save((flash_dataset.features, flash_dataset.targets), output_path)
            print(f"Optimized dataset saved at: {output_path}")
        
        return flash_dataset

    @staticmethod
    def from_pytorch_dataset_batched(
        pytorch_dataset: torch.utils.data.Dataset,
        memory_optimized: bool = True,
        output_path: str = None,
        batch_size: int = 64,
        max_ram_usage: float = 0.7,
        labels: Optional[List[str]] = None
    ) -> FlashDataset:
        import psutil
        import gc

        print(f"Converting PyTorch dataset to FlashDataset format (batch mode, batch_size={batch_size})...")
        
        total_ram = psutil.virtual_memory().total
        available_ram = psutil.virtual_memory().available
        max_allowed_ram = total_ram * max_ram_usage
        print(f"   Total RAM: {total_ram / (1024**3):.2f} GB")
        print(f"   Available RAM: {available_ram / (1024**3):.2f} GB")
        print(f"   RAM limit for conversion: {max_allowed_ram / (1024**3):.2f} GB")

        temp_loader = torch.utils.data.DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        all_features_list = []
        all_targets_list = []
        total_samples = len(pytorch_dataset)
        processed_samples = 0

        for batch_idx, (batch_features, batch_targets) in enumerate(tqdm(temp_loader, desc="Processing batches")):
            if not isinstance(batch_features, torch.Tensor):
                 batch_features = torch.tensor(batch_features)
            if not isinstance(batch_targets, torch.Tensor):
                 batch_targets = torch.tensor(batch_targets)

            all_features_list.append(batch_features)
            all_targets_list.append(batch_targets)

            processed_samples += len(batch_features)

        print(f"\n   Concatenating {len(all_features_list)} feature batches and {len(all_targets_list)} target batches...")
        if all_features_list:
            all_features = torch.cat(all_features_list, dim=0)
        else:
            all_features = torch.empty((0,))

        if all_targets_list:
            all_targets = torch.cat(all_targets_list, dim=0)
        else:
            all_targets = None

        del all_features_list, all_targets_list
        gc.collect()

        print(f"   Final features tensor shape: {all_features.shape}")
        if all_targets is not None:
            print(f"   Final targets tensor shape: {all_targets.shape}")

        if all_targets is not None:
            data_source = (all_features, all_targets)
        else:
            data_source = all_features
            
        flash_dataset = FlashDataset(
            data_source=data_source,
            memory_optimized=memory_optimized,
            labels=labels
        )

        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            flash_dataset.save(output_path)
            print(f"Optimized dataset saved at: {output_path}")
        
        print(f"✅ Batch conversion completed.")
        return flash_dataset

    @staticmethod
    def from_numpy(
        features: np.ndarray,
        targets: np.ndarray = None,
        memory_optimized: bool = True
    ) -> FlashDataset:
        if targets is not None:
            data = (features, targets)
        else:
            data = features
        
        return FlashDataset(data_source=data, memory_optimized=memory_optimized)
    
    @staticmethod
    def create_chunked_dataset(
        pytorch_dataset: torch.utils.data.Dataset,
        output_dir: str,
        chunk_size: int = 10000,
        batch_size: int = 64,
        memory_optimized: bool = True,
        labels: Optional[List[str]] = None,
        max_ram_usage: float = 0.7,
        shuffle: bool = False,
        split_ratios: Optional[Dict[str, float]] = None,
        feature_key: Optional[str] = None,
        target_key: Optional[str] = None
    ) -> str:
        import psutil
        import gc
        import random
        
        if split_ratios is not None:
            if abs(sum(split_ratios.values()) - 1.0) > 1e-5:
                raise ValueError(f"Split ratios must sum to 1.0. Current sum: {sum(split_ratios.values())}")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"📦 Creating chunked dataset at: {output_dir}")
        print(f"   Configuration: shuffle={shuffle}, splits={split_ratios}")
        if feature_key:
            print(f"   Dictionary Mode: feature_key='{feature_key}', target_key='{target_key}'")
        
        total_ram = psutil.virtual_memory().total
        max_allowed_ram = total_ram * max_ram_usage
        print(f"   Total RAM: {total_ram / (1024**3):.2f} GB")
        print(f"   RAM limit: {max_allowed_ram / (1024**3):.2f} GB")
        
        has_targets = True
        try:
            first_sample = pytorch_dataset[0]
            if feature_key:
                if feature_key not in first_sample:
                    raise KeyError(f"Feature key '{feature_key}' not found in first sample.")
                if target_key and target_key not in first_sample:
                    print(f"⚠️ Warning: Target key '{target_key}' not found. Assuming dataset without targets.")
                    has_targets = False
                elif target_key is None:
                    has_targets = False
            else:
                has_targets = isinstance(first_sample, tuple) and len(first_sample) == 2
        except Exception as e:
            print(f"⚠️ Warning: Could not inspect first sample: {e}")
        
        current_chunk_features = []
        current_chunk_targets = []
        current_chunk_idx = 0
        total_samples = len(pytorch_dataset)
        samples_processed = 0
        samples_skipped = 0
        
        feature_shape = None
        target_shape = None
        feature_dtype = None
        target_dtype = None
        chunk_indices = []
        
        indices = list(range(total_samples))
        if shuffle:
            print("🎲 Shuffling dataset indices...")
            random.shuffle(indices)
        
        print(f"🔄 Processing {total_samples} samples in chunks of {chunk_size}...")
        
        for i in tqdm(indices, desc="Processing dataset"):
            try:
                sample = pytorch_dataset[i]
                
                feature = None
                target = None
                
                if feature_key:
                    feature = sample[feature_key]
                    if has_targets and target_key:
                        target = sample[target_key]
                elif has_targets and isinstance(sample, tuple):
                    feature, target = sample
                else:
                    feature = sample
                
                if not isinstance(feature, torch.Tensor):
                    feature = torch.tensor(feature)
                
                current_chunk_features.append(feature)
                
                if has_targets and target is not None:
                    if not isinstance(target, torch.Tensor):
                        target = torch.tensor(target)
                    current_chunk_targets.append(target)
            
                if feature_shape is None:
                    feature_shape = tuple(current_chunk_features[-1].shape)
                    feature_dtype = str(current_chunk_features[-1].dtype)
                
                if target_shape is None and has_targets and len(current_chunk_targets) > 0:
                    target_shape = tuple(current_chunk_targets[-1].shape) if hasattr(current_chunk_targets[-1], 'shape') else (1,)
                    target_dtype = str(current_chunk_targets[-1].dtype)
                
                samples_processed += 1
            
            except UnidentifiedImageError as e:
                print(f"\n⚠️  Warning: Error identifying image at index {i}: {e}")
                samples_skipped += 1
                continue
            except Exception as e:
                print(f"\n⚠️  Warning: Unexpected error processing sample at index {i}: {e}")
                samples_skipped += 1
                continue
            
            if len(current_chunk_features) >= chunk_size:
                chunk_start_idx = current_chunk_idx * chunk_size
                chunk_end_idx = chunk_start_idx + len(current_chunk_features)
                chunk_indices.append((chunk_start_idx, chunk_end_idx))
                
                DatasetConverter._save_chunk(
                    current_chunk_features,
                    current_chunk_targets,
                    output_dir,
                    current_chunk_idx,
                    memory_optimized
                )
                
                print(f"💾 Chunk {current_chunk_idx} saved ({len(current_chunk_features)} samples)")
                
                current_chunk_features = []
                current_chunk_targets = []
                current_chunk_idx += 1
                
                gc.collect()
        
        print(f"\n📊 Conversion Summary:")
        print(f"   Successfully processed samples: {samples_processed}")
        print(f"   Samples skipped due to error: {samples_skipped}")
        print(f"   Total samples in original dataset: {len(pytorch_dataset)}")
        
        if current_chunk_features:
            chunk_start_idx = current_chunk_idx * chunk_size
            chunk_end_idx = chunk_start_idx + len(current_chunk_features)
            chunk_indices.append((chunk_start_idx, chunk_end_idx))
            
            DatasetConverter._save_chunk(
                current_chunk_features,
                current_chunk_targets,
                output_dir,
                current_chunk_idx,
                memory_optimized
            )
            print(f"💾 Last chunk {current_chunk_idx} saved ({len(current_chunk_features)} samples)")
            current_chunk_idx += 1
        
        num_chunks = current_chunk_idx
        print(f"   Total chunks generated: {num_chunks}")

        splits_mapping = {}
        if split_ratios is not None:
            total_chunks = num_chunks
            num_train = int(total_chunks * split_ratios.get('train', 0.8))
            num_val = int(total_chunks * split_ratios.get('val', 0.1))
            num_test = total_chunks - num_train - num_val
            
            all_chunk_indices = list(range(total_chunks))
            
            splits_mapping['train'] = all_chunk_indices[:num_train]
            splits_mapping['val'] = all_chunk_indices[num_train:num_train+num_val]
            splits_mapping['test'] = all_chunk_indices[num_train+num_val:]
            
            print(f"✂️  Chunk Distribution:")
            print(f"   Train: {len(splits_mapping['train'])} chunks")
            print(f"   Val:   {len(splits_mapping['val'])} chunks")
            print(f"   Test:  {len(splits_mapping['test'])} chunks")
        else:
             print(f"ℹ️  No splits configured (split_ratios=None). All chunks will be available.")
        
        metadata = {
            'total_samples': samples_processed,
            'feature_shape': feature_shape,
            'target_shape': target_shape,
            'feature_dtype': feature_dtype,
            'target_dtype': target_dtype,
            'num_chunks': num_chunks,
            'chunk_indices': chunk_indices,
            'chunk_size': chunk_size,
            'labels': labels,
            'original_dataset_size': len(pytorch_dataset),
            'samples_skipped': samples_skipped,
            'splits': splits_mapping,
            'split_ratios': split_ratios
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Metadata saved at: {metadata_path}")
        print(f"🎉 Chunked dataset created successfully!")
        print(f"   Directory: {output_dir}")
        
        return output_dir
    
    @staticmethod
    def _save_chunk(features, targets, output_dir, chunk_idx, memory_optimized):
        features_tensor = torch.stack(features)
        targets_tensor = torch.stack(targets) if targets else None
        
        if memory_optimized:
            from .memory import MemoryOptimizer
            mem_optimizer = MemoryOptimizer()
            features_tensor = mem_optimizer.optimize_tensor(features_tensor)
            if targets_tensor is not None:
                targets_tensor = mem_optimizer.optimize_tensor(targets_tensor)
        
        chunk_path = os.path.join(output_dir, f'chunk_{chunk_idx}.pt')
        torch.save({
            'features': features_tensor,
            'targets': targets_tensor
        }, chunk_path)