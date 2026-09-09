from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import cv2
import pickle
from PIL import Image
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler
import tqdm
from collections import deque
import yaml 

from yolo.data.dataset_utils import (
    scale,
    denormalize,
    normalize,
    add_padding)



from yolo.config import SimplifiedConfig

__all__ = ["YoloDataset", "create_dataloaders", "DatasetSplit"]


@dataclass()
class DatasetSplit:
    images: List[Path]
    labels: List[Path]
    class_names: Dict[int, str]


def load_data_split(data_yaml: Path, split: str) -> DatasetSplit:
    """Load dataset split from YAML file (YOLO format)."""


    with data_yaml.open('r') as f:
        data_dict = yaml.safe_load(f)
    
    if split not in data_dict:
        raise KeyError(f"Dataset split '{split}' missing from {data_yaml}")

    yaml_path = data_dict.get("path")
    if yaml_path:
        root = Path(yaml_path)
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
    else:
        root = data_yaml.parent
    image_dir = Path(data_dict[split])
    if not image_dir.is_absolute():
        image_dir = (root / image_dir).resolve()
    cache_filename = f"simplified_labels_{image_dir.stem}.cache"
    cache_path = image_dir.parent / cache_filename
    if cache_path.exists():
        with cache_path.open("rb") as f:
            data_dict = pickle.load(f)
            num_classes = data_dict["nc"]
        print(f"Using cached labels from {cache_path}")
    else:
        data_dict, num_classes = verify_data(data_dict, image_dir, cache_path, split=split, root=root)

    return data_dict, num_classes

def verify_data(
    data_dict: Dict,
    image_dir: Path,
    cache_path: Path,
    split: str | None = None,
    root: Path | None = None,
) -> Tuple[Dict, int]:
    """
    Verify and cache dataset for object detection.

    `image_dir` may be either:
      - a directory containing image files (labels resolved via sibling `labels/<split>` dir), or
      - a .txt list file where each line is a path to an image,
        relative to `root`. Labels are derived by replacing `/images/` with
        `/labels/` and the extension with `.txt`.
    """
    num_classes = data_dict.get("nc")
    if num_classes is None:
        names = data_dict.get("names")
        if isinstance(names, dict):
            num_classes = len(names)
        elif isinstance(names, (list, tuple)):
            num_classes = len(names)
        else:
            num_classes = 0
    num_classes = int(num_classes)
    IMG_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if image_dir.is_file() and image_dir.suffix.lower() == ".txt":
        if root is None:
            root = image_dir.parent
        with image_dir.open() as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        image_files: List[Path] = []
        for ln in lines:
            p = Path(ln)
            if not p.is_absolute():
                p = (root / ln).resolve()
            if p.suffix.lower() in IMG_FORMATS:
                image_files.append(p)
        if not image_files:
            raise ValueError(f"No image paths found in list file: {image_dir}")
        sep = "/"

        def label_for(img_path: Path) -> Path:
            s = str(img_path).replace("\\", "/")
            token = f"{sep}images{sep}"
            idx = s.rfind(token)
            if idx >= 0:
                s = s[:idx] + f"{sep}labels{sep}" + s[idx + len(token):]
            return Path(s).with_suffix(".txt")

        print(f"The image list file is: {image_dir}")
        print(f"Resolved {len(image_files)} image paths (root={root})")
        split_name = image_dir.stem
    elif image_dir.is_dir():
        labels_dir_candidates = []
        split_name = image_dir.name
        parent = image_dir.parent
        grandparent = parent.parent if parent else None

        labels_dir_candidates.append(parent / "labels" / split_name)
        if grandparent:
            labels_dir_candidates.append(grandparent / "labels" / split_name)
        labels_dir_candidates.append(parent / "labels")
        if grandparent:
            labels_dir_candidates.append(grandparent / "labels")

        labels_dir = next(
            (path for path in labels_dir_candidates if path and path.exists() and path.is_dir()),
            None,
        )

        if labels_dir is None:
            raise FileNotFoundError(
                f"Labels directory not found for split '{split_name}'."
                f" Checked: {', '.join(str(c) for c in labels_dir_candidates)}"
            )

        print(f"The image directory is: {image_dir}")
        print(f"The labels directory is: {labels_dir}")
        print(f"The image_dir.name is: {split_name}")

        image_files = [
            f for f in sorted(image_dir.iterdir())
            if f.suffix.lower() in IMG_FORMATS
        ]

        def label_for(img_path: Path, _ld=labels_dir) -> Path:
            return _ld / f"{img_path.stem}.txt"
    else:
        raise FileNotFoundError(f"Image path not found or unsupported: {image_dir}")

    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    # Initialize counters
    corrupt_count = 0
    background_count = 0
    valid_count = 0

    data = {"labels": []}

    print(f"\nVerifying {len(image_files)} images for split '{split_name}'")
    class_index_seen_dict = {}
    # Process each image
    for img_path in tqdm.tqdm(image_files, desc="Scanning"):
        label_path = label_for(img_path)
        
        try:
            # Verify image.
            img = Image.open(img_path)
            img.verify()  # Check if image is corrupt
            
            # Reopen to get size (verify() invalidates the image)
            img = Image.open(img_path)
            img_shape = (img.height, img.width)  # (height, width)
            
            # Validate dimensions
            if img_shape[0] < 10 or img_shape[1] < 10:
                print(f"  ⚠️  Skipping {img_path.name}: image too small {img_shape}")
                corrupt_count += 1
                continue

            if label_path.exists():
                with label_path.open() as f:
                    lines = [
                        x.split() for x in f.read().strip().splitlines() 
                        if len(x.strip()) > 0
                    ]
                
                if len(lines) > 0:
                    labels = np.array(lines, dtype=np.float32)
                    
                    # Validate shape (must be [N, 5]: class, x, y, w, h)
                    if labels.shape[1] != 5:
                        print(f"Skipping {img_path.name}: expected 5 columns, got {labels.shape[1]}")
                        corrupt_count += 1
                        continue
                    
                    # Validate normalization [0, 1]
                    # [x, y, w, h]
                    coords = labels[:, 1:]  
                    if coords.max() > 1.0 or coords.min() < 0.0:
                        print(f"Skipping {img_path.name}: coordinates not normalized")
                        corrupt_count += 1
                        continue
                    
                    # Validate class indices
                    classes = labels[:, 0]
                    #if (classes >= num_classes).any() or (classes < 0).any():
                    if (classes < 0).any() or (num_classes > 0 and max(classes) >= num_classes):
                        invalid_cls = classes[(classes >= num_classes) | (classes < 0)]
                        print(f"Skipping {img_path.name}: invalid class indices {invalid_cls}")
                        corrupt_count += 1
                        continue
                    # Track seen class indices
                    for cls in classes:
                        cls_int = int(cls)
                        if cls_int not in class_index_seen_dict:
                            class_index_seen_dict[cls_int] = 1
                        else:
                            class_index_seen_dict[cls_int] += 1
                    
                    # Remove duplicates
                    unique_labels, _ = np.unique(labels, axis=0, return_index=True)
                    if len(unique_labels) < len(labels):
                        print(f" {img_path.name}: removed {len(labels) - len(unique_labels)} duplicate labels")
                        labels = unique_labels
                    
                    valid_count += 1
                
                else:
                    # Background file found
                    labels = np.zeros((0, 5), dtype=np.float32)
                    background_count += 1
            
            else:
                # No label file found, default to background
                labels = np.zeros((0, 5), dtype=np.float32)
                background_count += 1
            
            data["labels"].append({
                "img_file": str(img_path),
                "img_shape": img_shape,
                "cls": labels[:, 0:1],    
                "bboxes": labels[:, 1:],  
                "normalized": True,
            })
        
        except Exception as e:
            print(f" Skipping {img_path.name}: {e}")
            corrupt_count += 1
            continue
    if not class_index_seen_dict:
        raise ValueError(f"No labeled images found in split '{split_name}'. Cannot determine number of classes.")
    num_classes_seen = max(class_index_seen_dict.keys()) + 1
    cache_data = {
        "labels": data["labels"],
        "nc": num_classes_seen,
        "results": {
            "valid": valid_count,
            "background": background_count,
            "corrupt": corrupt_count,
            "total": len(image_files)
        },
        "names": data_dict.get("names"),
        "split": split,
    }
    
    # Save to disk
    with cache_path.open("wb") as f:
        pickle.dump(cache_data, f)
    
    print(f"\n{'='*60}")
    print(f"Verification results for '{image_dir.name}':")
    print(f"Valid images with labels: {valid_count}")
    print(f"Background images: {background_count}")
    print(f"Corrupt/invalid: {corrupt_count}")
    print(f"Total processed: {len(image_files)}")
    print(f"Dataset size: {len(data['labels'])} images")
    print(f"Cache saved to: {cache_path}")
    print(f"{'='*60}\n")
    
    
    return cache_data, num_classes_seen




class YoloDataset(Dataset):
    def __init__(
        self,
        dataset: Dict,
        imgsz: int,
        augment: bool = False,
        batch_size: int = 16,
        transforms=None,
        rect: bool = False,
        single_class: bool = False,
        classes_to_ignore: List[int] = None,
        do_letterbox: bool = True,
    ) -> None:
        self.dataset = dataset
        self.imgsz = imgsz
        self.do_augment = augment
        self.batch_size = batch_size
        self.transforms = transforms
         # can be toggled off by close_mosaic
        self.mosaic = True 
        self.do_letterbox = do_letterbox
        if rect and not do_letterbox:
            print("rect requires do_letterbox=True, disabling rect")
            rect = False
        self.rect = rect
        
        self.length_of_labels = len(dataset["labels"])
        self.update_labels(single_class, classes_to_ignore or [])

        self.indices = list(range(self.length_of_labels))
        self.batch_shapes = None

        # Buffer for mosaic augmentation
        self.max_buffer_length = min(self.length_of_labels, self.batch_size * 8, 1000) if self.do_augment else 0
        self.buffer = deque(maxlen=self.max_buffer_length)
        
        # Image caching
        self.buffer_image = {}
        self.image_orig_shape = {}
        self.image_resized_shape = {}
    
    def update_labels(
            self,
            single_class:bool,
            classes_to_ignore:List[int]):
        
        """Update labels to ignore specified classes or convert to single class."""
        
        for item in self.dataset["labels"]:
            if classes_to_ignore is not None and len(classes_to_ignore) > 0:
                cls = item["cls"]
                bboxes = item["bboxes"]
                indices = ~np.isin(cls, classes_to_ignore).squeeze(-1)
                cls = item["cls"][indices]
                bboxes = item["bboxes"][indices]
            

                item["cls"] = cls
                item["bboxes"] = bboxes
    
            if single_class:
                if len(item["cls"]) > 0:
                    item["cls"][:,0] = 0 



    def set_rect_shapes(self, batch_size: int, stride: int = 32, pad: float = 0.0):
        """Sort by aspect ratio and compute per-batch target shapes."""
        n = self.length_of_labels
        shapes = np.array([self.dataset["labels"][i]["img_shape"] for i in range(n)], dtype=np.float64)
         # h / w
        ar = shapes[:, 0] / shapes[:, 1] 

        sorted_order = ar.argsort()
        self.indices = sorted_order.tolist()

        bi = np.floor(np.arange(n) / batch_size).astype(int)
        num_batches = bi[-1] + 1
        self.batch_shapes = np.zeros((num_batches, 2), dtype=np.int64)

        ar_sorted = ar[sorted_order]
        for i in range(num_batches):
            ari = ar_sorted[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shape_ratio = np.array([maxi, 1], dtype=np.float64)
            elif mini > 1:
                shape_ratio = np.array([1, 1 / mini], dtype=np.float64)
            else:
                shape_ratio = np.array([1, 1], dtype=np.float64)

            self.batch_shapes[i] = (np.ceil(shape_ratio * self.imgsz / stride + pad) * stride).astype(np.int64)

    @property
    def ni(self):
        """Number of images in dataset."""
        return self.length_of_labels
    
    def load_image(self, index):
        """Load and optionally cache image."""
        # Check cache first
        if index in self.buffer_image:
            return self.buffer_image[index], self.image_orig_shape[index], self.image_resized_shape[index]
        
        image_path = self.dataset["labels"][index]["img_file"]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        
        orig_h, orig_w = image.shape[:2]

        if self.do_letterbox:
            r = self.imgsz / max(orig_h, orig_w)
            if r != 1:
                interp = cv2.INTER_LINEAR if (self.do_augment or r > 1) else cv2.INTER_AREA
                image = cv2.resize(image, (math.ceil(orig_w * r), math.ceil(orig_h * r)), interpolation=interp)
        else:
            interp = cv2.INTER_LINEAR if self.do_augment else cv2.INTER_AREA
            image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=interp)
        new_h, new_w = image.shape[:2]
        
        # Cache if augmenting
        if self.do_augment:
            self.buffer_image[index] = image
            self.image_orig_shape[index] = (orig_h, orig_w)
            self.image_resized_shape[index] = (new_h, new_w)
            self.buffer.append(index)
            
            # Remove oldest cached image if buffer full
            if len(self.buffer) >= self.max_buffer_length:
                old_idx = self.buffer[0]
                if old_idx in self.buffer_image:
                    del self.buffer_image[old_idx]
                    del self.image_orig_shape[old_idx]
                    del self.image_resized_shape[old_idx]
        
        return image, (orig_h, orig_w), (new_h, new_w)

    def _get_label_and_image(self, index: int) -> Dict:
        """Get image and label data for a specific index."""
        label = deepcopy(self.dataset["labels"][index])
        label["image"], label["orig_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio"] = (
            label["resized_shape"][0] / label["orig_shape"][0], 
            label["resized_shape"][1] / label["orig_shape"][1]
        )
        label["sample_idx"] = index
        return label

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get transformed item."""
        index = self.indices[idx]
        data = self._get_label_and_image(index)
        
        if self.rect and self.do_letterbox and self.batch_shapes is not None:
            batch_idx = idx // self.batch_size
            data["rect_shape"] = tuple(self.batch_shapes[batch_idx])
        
        if self.transforms is not None:
            data = self.transforms(data)
        
        return data

    def close_mosaic(self, hyp):
        """Rebuild transforms with mosaic disabled (called for last N epochs)."""
        from .augmentations import build_transforms
        hyp = dict(hyp) if not isinstance(hyp, dict) else hyp
        hyp['mosaic'] = 0.0
        self.transforms = build_transforms(
            dataset=self,
            imgsz=self.imgsz,
            hyp=hyp,
        )
        self.mosaic = False



def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def load_image_shape(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.height, im.width


def load_labels(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 5), dtype=np.float32)

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return np.zeros((0, 5), dtype=np.float32)

    entries: List[List[float]] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Label file '{path}' contains malformed line: '{line}'")
        try:
            entries.append([float(value) for value in parts])
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Invalid numeric value in label file '{path}': '{line}'") from exc

    return np.array(entries, dtype=np.float32)



def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Combines individual samples into a batch, handling variable-sized bboxes.
    """
    images = torch.stack([item["img"] for item in batch], dim=0)
    
    # Combine all targets with batch index
    targets = []
    for i, item in enumerate(batch):
        labels = item.get("bboxes", torch.zeros((0, 4)))
        cls = item.get("cls", torch.zeros((0, 1)))
        
        if labels.numel() > 0:
            # Concatenate: [batch_idx, class, x, y, w, h]
            batch_idx = torch.full((labels.shape[0], 1), i, dtype=labels.dtype)
            target = torch.cat((batch_idx, cls, labels), dim=1)
            targets.append(target)
    
    targets = torch.cat(targets, dim=0) if targets else torch.zeros((0, 6), dtype=images.dtype)
    
    return {
        "images": images,
        "targets": targets,
    }


def create_dataloaders(config: SimplifiedConfig,
                       shuffle_val: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration object with dataset and training settings
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from .augmentations import build_transforms
    
    data_yaml = Path(config.data)
    train_split, num_classes = load_data_split(data_yaml, "train")
    val_split,_ = load_data_split(data_yaml, "val")
    if getattr(config, 'single_class', False):
        config.nc = 1
    # Build augmentation hyperparameters from config
    train_hyp = config.get_augmentation_hyp() if hasattr(config, 'get_augmentation_hyp') else {
        'mosaic': getattr(config, 'mosaic', 1.0),
        'do_letterbox': True,
        'hsv_h': getattr(config, 'hsv_h', 0.015),
        'hsv_s': getattr(config, 'hsv_s', 0.7),
        'hsv_v': getattr(config, 'hsv_v', 0.4),
        'flipud': getattr(config, 'flipud', 0.0),
        'fliplr': getattr(config, 'fliplr', 0.5),
        'random_affine': getattr(config, 'random_affine', False),
    }
    
    # Validation transforms (no augmentation)
    val_hyp = {
        'mosaic': 0.0,
        'do_letterbox': True,
        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'flipud': 0.0,
        'fliplr': 0.0,
        'random_affine': False,
    }
    
    do_letterbox = getattr(config, "do_letterbox", True)
    rect_training = getattr(config, "rect", False)
    if rect_training and not do_letterbox:
        print("rect requires do_letterbox=True, disabling rect")
        rect_training = False

    train_hyp['do_letterbox'] = do_letterbox
    val_hyp['do_letterbox'] = do_letterbox
    train_hyp.setdefault('rect', rect_training)
    val_hyp.setdefault('rect', rect_training)

    classes_to_ignore = getattr(config, 'classes_to_ignore', None) or []
    single_class = getattr(config, 'single_class', False)

    # Create datasets first (without transforms)
    train_dataset = YoloDataset(
        train_split,
        imgsz=config.imgsz,
        augment=True,
        batch_size=config.batch_size,
        transforms=None,
        rect=rect_training,
        classes_to_ignore=classes_to_ignore,
        single_class=single_class,
        do_letterbox=do_letterbox,
    )

    val_dataset = YoloDataset(
        val_split,
        imgsz=config.imgsz,
        augment=False,
        batch_size=config.batch_size,
        transforms=None,  
        rect=rect_training,
        classes_to_ignore=classes_to_ignore,
        single_class=single_class,
        do_letterbox=do_letterbox,
    )
    # Build transforms with dataset reference
    train_transforms = build_transforms(
        dataset=train_dataset,
        imgsz=config.imgsz,
        hyp=train_hyp,
        rect=rect_training
    )
    
    val_transforms = build_transforms(
        dataset=val_dataset,
        imgsz=config.imgsz,
        hyp=val_hyp,
        rect=rect_training
    )
    
    # Set transforms on datasets
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms

    if rect_training:
        train_dataset.set_rect_shapes(config.batch_size, pad=0.0)
        val_dataset.set_rect_shapes(config.batch_size, pad=0.5)

    persistent = config.workers > 0

    def worker_init(_):
        cv2.setNumThreads(0)

    if rect_training:
        train_sampler = BatchSampler(
            SequentialSampler(train_dataset),
            batch_size=config.batch_size,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=config.workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=persistent,
            worker_init_fn=worker_init,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.workers,
            pin_memory=False,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=persistent,
            worker_init_fn= worker_init,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle_val,
        num_workers=config.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=False,
        worker_init_fn= worker_init,
    )

    return train_loader, val_loader

