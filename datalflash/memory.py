# datalflash/memory.py
import torch
import numpy as np
import psutil
from typing import Any, Union

class MemoryOptimizer:
    def __init__(self):
        self.available_memory = psutil.virtual_memory().available
        self.gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    
    def optimize_tensor(
        self,
        tensor: Union[torch.Tensor, np.ndarray, list],
        target_dtype: torch.dtype = torch.float16,
        use_quantization: bool = True
    ) -> torch.Tensor:
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        elif isinstance(tensor, list):
            tensor = torch.tensor(tensor)
        elif not isinstance(tensor, torch.Tensor):
            raise ValueError("Tensor type not supported")
        
        if use_quantization:
            if tensor.dtype in [torch.float32, torch.float64]:
                tensor = tensor.to(target_dtype)
        
        tensor = tensor.contiguous()
        
        return tensor
    
    def get_optimal_batch_size(
        self,
        sample_size_bytes: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> int:
        if device == 'cuda' and self.gpu_memory > 0:
            available_mem = self.gpu_memory * 0.8
        else:
            available_mem = self.available_memory * 0.6
        
        min_batch_size = max(1, int(available_mem / (sample_size_bytes * 10)))
        
        optimal_batch_size = int(available_mem / sample_size_bytes)
        
        return min(max(1, min_batch_size), optimal_batch_size)
    
    def enable_memory_efficient_mode(self):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True