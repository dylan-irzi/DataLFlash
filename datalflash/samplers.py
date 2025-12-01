import torch
import math
from typing import Iterator, Optional, Sequence, List, Sized

class Sampler:
    def __init__(self, data_source: Optional[Sized]):
        pass

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

class SequentialSampler(Sampler):
    def __init__(self, data_source: Sized):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)

class RandomSampler(Sampler):
    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got {type(self.replacement)}")

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since it is always len(data_source).")

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples):
                yield torch.randint(high=n, size=(1,), dtype=torch.int64, generator=generator).item()
        else:
            for i in torch.randperm(n, generator=generator):
                yield i.item()

    def __len__(self) -> int:
        return self.num_samples

class BatchSampler(Sampler):
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got {batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got {drop_last}")
        
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
