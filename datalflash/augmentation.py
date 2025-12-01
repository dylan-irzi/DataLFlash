# datalflash/augmentation.py
import torch
import torch.nn.functional as F
import numpy as np
import numbers
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class BatchTransform:
    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class Compose(BatchTransform):
    def __init__(self, transforms: List[BatchTransform]):
        self.transforms = transforms

    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for t in self.transforms:
            if isinstance(t, BatchTransform):
                features, targets = t(features, targets)
            else:
                features = t(features)
        return features, targets

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class MixUp(BatchTransform):
    def __init__(self, alpha: float = 1.0, prob: float = 0.5, num_classes: Optional[int] = None):
        self.alpha = alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if np.random.rand() > self.prob:
            return features, targets

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = features.size(0)
        index = torch.randperm(batch_size).to(features.device)

        mixed_features = lam * features + (1 - lam) * features[index]
        
        if targets is not None:
            if not targets.is_floating_point():
                if self.num_classes is not None:
                    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
                    mixed_targets = lam * targets_one_hot + (1 - lam) * targets_one_hot[index]
                else:
                    mixed_targets = lam * targets.float() + (1 - lam) * targets[index].float()
            else:
                mixed_targets = lam * targets + (1 - lam) * targets[index]
        else:
            mixed_targets = None

        return mixed_features, mixed_targets

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, prob={self.prob}, num_classes={self.num_classes})"

class CutMix(BatchTransform):
    def __init__(self, alpha: float = 1.0, prob: float = 0.5, num_classes: Optional[int] = None):
        self.alpha = alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if np.random.rand() > self.prob:
            return features, targets

        batch_size, _, h, w = features.size()
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (features.size()[-1] * features.size()[-2]))

        index = torch.randperm(batch_size).to(features.device)
        
        features = features.clone()
        features[:, :, bby1:bby2, bbx1:bbx2] = features[index, :, bby1:bby2, bbx1:bbx2]

        if targets is not None:
            if not targets.is_floating_point():
                if self.num_classes is not None:
                    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
                    mixed_targets = lam * targets_one_hot + (1 - lam) * targets_one_hot[index]
                else:
                    mixed_targets = lam * targets.float() + (1 - lam) * targets[index].float()
            else:
                mixed_targets = lam * targets + (1 - lam) * targets[index]
        else:
            mixed_targets = None
            
        return features, mixed_targets

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, prob={self.prob}, num_classes={self.num_classes})"

class RandomHorizontalFlip(BatchTransform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if np.random.rand() < self.p:
            return torch.flip(features, [-1]), targets
        return features, targets

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

class RandomVerticalFlip(BatchTransform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if np.random.rand() < self.p:
            return torch.flip(features, [-2]), targets
        return features, targets

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

class GaussianNoise(BatchTransform):
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std

    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        noise = torch.randn_like(features) * self.std + self.mean
        return features + noise, targets

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class Normalize(BatchTransform):
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)

    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.mean.device != features.device:
            self.mean = self.mean.to(features.device)
            self.std = self.std.to(features.device)
            
        return (features - self.mean) / self.std, targets

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean.view(-1).tolist()}, std={self.std.view(-1).tolist()})"

class RandomBrightnessContrast(BatchTransform):
    def __init__(self, brightness_limit: float = 0.2, contrast_limit: float = 0.2, p: float = 0.5):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.p = p

    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if np.random.rand() < self.p:
            alpha = 1.0 + np.random.uniform(-self.brightness_limit, self.brightness_limit)
            features = features * alpha
            
            beta = 1.0 + np.random.uniform(-self.contrast_limit, self.contrast_limit)
            mean = features.mean(dim=(2, 3), keepdim=True) if features.ndim == 4 else features.mean()
            features = (features - mean) * beta + mean
            
        return features, targets

    def __repr__(self):
        return f"{self.__class__.__name__}(brightness_limit={self.brightness_limit}, contrast_limit={self.contrast_limit}, p={self.p})"

class RandomCrop(BatchTransform):
    def __init__(self, size: int, padding: int = 0, pad_if_needed: bool = False, fill: float = 0.0, p: float = 1.0):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.p = p

    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if np.random.rand() > self.p:
            return features, targets

        batch_size, channels, height, width = features.shape

        if self.padding > 0:
            features = F.pad(features, (self.padding, self.padding, self.padding, self.padding), value=self.fill)

        padded_height, padded_width = features.shape[-2], features.shape[-1]

        crop_h, crop_w = self.size
        if self.pad_if_needed:
            if padded_height < crop_h:
                features = F.pad(features, (0, 0, 0, crop_h - padded_height), value=self.fill)
                padded_height = crop_h
            if padded_width < crop_w:
                features = F.pad(features, (0, crop_w - padded_width, 0, 0), value=self.fill)
                padded_width = crop_w

        max_h = padded_height - crop_h
        max_w = padded_width - crop_w

        if max_h <= 0 or max_w <= 0:
            top = max(0, max_h // 2)
            left = max(0, max_w // 2)
        else:
            top = np.random.randint(0, max_h + 1)
            left = np.random.randint(0, max_w + 1)

        cropped_features = features[:, :, top:top + crop_h, left:left + crop_w]

        return cropped_features, targets

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding}, pad_if_needed={self.pad_if_needed}, fill={self.fill}, p={self.p})"

class ColorJitter(BatchTransform):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=1.0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.p = p

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non-negative.")
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if value[0] == value[1] == center:
            value = None
        return value

    def adjust_saturation(self, img, saturation_factor):
        if saturation_factor == 1:
            return img
        mean = img.mean(dim=1, keepdim=True)
        return (1 - saturation_factor) * mean + saturation_factor * img

    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if np.random.rand() > self.p:
            return features, targets

        transforms_order = [self._adjust_brightness, self._adjust_contrast, self._adjust_saturation]
        np.random.shuffle(transforms_order)

        for transform in transforms_order:
            features = transform(features)

        return features, targets

    def _adjust_brightness(self, features):
        if self.brightness is None:
            return features
        brightness_factor = np.random.uniform(self.brightness[0], self.brightness[1])
        return features * brightness_factor

    def _adjust_contrast(self, features):
        if self.contrast is None:
            return features
        contrast_factor = np.random.uniform(self.contrast[0], self.contrast[1])
        mean = features.mean(dim=[2, 3], keepdim=True)
        return (features - mean) * contrast_factor + mean

    def _adjust_saturation(self, features):
        if self.saturation is None:
            return features
        saturation_factor = np.random.uniform(self.saturation[0], self.saturation[1])
        return self.adjust_saturation(features, saturation_factor)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'brightness={self.brightness}'
        format_string += f', contrast={self.contrast}'
        format_string += f', saturation={self.saturation}'
        format_string += f', hue={self.hue}'
        format_string += f', p={self.p}'
        format_string += ')'
        return format_string

class Cutout(BatchTransform):
    def __init__(self, size: int, p: float = 0.5, fill: float = 0.0):
        self.size = size
        self.p = p
        self.fill = fill

    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if np.random.rand() > self.p:
            return features, targets

        batch_size, channels, height, width = features.shape

        center_h = torch.randint(0, height, size=(batch_size,), device=features.device)
        center_w = torch.randint(0, width, size=(batch_size,), device=features.device)

        half_size = self.size // 2
        y1 = torch.clamp(center_h - half_size, 0, height)
        y2 = torch.clamp(center_h + half_size, 0, height)
        x1 = torch.clamp(center_w - half_size, 0, width)
        x2 = torch.clamp(center_w + half_size, 0, width)

        h_idx = torch.arange(height, device=features.device)[None, None, :, None]
        w_idx = torch.arange(width, device=features.device)[None, None, None, :]

        y1_expanded = y1[:, None, None, None]
        y2_expanded = y2[:, None, None, None]
        x1_expanded = x1[:, None, None, None]
        x2_expanded = x2[:, None, None, None]

        mask = (h_idx >= y1_expanded) & (h_idx < y2_expanded) & \
               (w_idx >= x1_expanded) & (w_idx < x2_expanded)

        features = features.clone()
        features[mask.expand_as(features)] = self.fill

        return features, targets

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, p={self.p}, fill={self.fill})"
        
class RandomAffine(BatchTransform):
    def __init__(self, degrees=0, translate=None, scale=None, shear=None, p=0.5):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p

    def __call__(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if np.random.rand() > self.p:
            return features, targets

        batch_size, channels, height, width = features.shape

        try:
            import kornia
            print("INFO: RandomAffine using Kornia is not fully implemented in this code example.")
            return features, targets
        except ImportError:
            print("WARNING: Kornia is not installed. RandomAffine requires Kornia for efficient batch implementation.")
            return features, targets

    def __repr__(self):
        return f"{self.__class__.__name__}(degrees={self.degrees}, translate={self.translate}, scale={self.scale}, shear={self.shear}, p={self.p})"