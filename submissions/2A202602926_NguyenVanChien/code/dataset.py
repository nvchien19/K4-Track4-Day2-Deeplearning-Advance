"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Chạy trên Colab/Kaggle (có torch + torchvision). Giữ đúng giao diện starter/:
    load_split / check_split / build_transforms / DeepWeedsDataset / make_loader
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
import numpy as np  # noqa: E402  (torch truoc numpy: tranh xung dot OpenMP/MKL tren Windows)
import pandas as pd  # noqa: E402
from PIL import Image

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train/val/test_subset{fold}.csv, không sửa/lọc/chia lại (S1)."""
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if not {"Filename", "Label"} <= set(df.columns):
            raise ValueError(f"{name}: cần cột Filename và Label, có {list(df.columns)}")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc README 2.1: số ảnh, per-class, giao rỗng, hợp 17509, file tồn tại."""
    images_dir = Path(images_dir)
    n = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    total = sum(n.values())
    per_class = {}
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        vc = df["Label"].value_counts().sort_index()
        per_class[name] = {int(k): int(v) for k, v in vc.items()}
    s_tr, s_va, s_te = set(train_df["Filename"]), set(val_df["Filename"]), set(test_df["Filename"])
    overlap = {
        "train_val": len(s_tr & s_va),
        "train_test": len(s_tr & s_te),
        "val_test": len(s_va & s_te),
    }
    assert overlap["train_val"] == 0, f"train∩val = {overlap['train_val']} (phải rỗng)"
    assert overlap["train_test"] == 0, f"train∩test = {overlap['train_test']} (phải rỗng)"
    assert overlap["val_test"] == 0, f"val∩test = {overlap['val_test']} (phải rỗng)"
    assert total == TOTAL_IMAGES, f"hợp 3 tập = {total}, kỳ vọng {TOTAL_IMAGES}"
    for name, exp in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        ratio = n[name] / total
        assert abs(ratio - exp) <= 0.01 + 1e-9, f"tỉ lệ {name}={ratio:.4f}, lệch >1pp so với {exp}"
    missing = [f for f in (s_tr | s_va | s_te) if not (images_dir / f).exists()]
    assert not missing, f"thiếu {len(missing)} file ảnh trong {images_dir}, vd: {missing[:5]}"
    print(f"split OK: train={n['train']} val={n['val']} test={n['test']} total={total}")
    print(f"overlap: {overlap} | union={len(s_tr | s_va | s_te)}")
    return {"n": n, "per_class": per_class, "overlap": overlap, "total": total}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Train: RandomResizedCrop + hflip (+color nếu aug>=color); Val/test: Resize256+CenterCrop, không aug ngẫu nhiên."""
    from torchvision import transforms
    normalize = transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
    if not train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])
    tf = [transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
          transforms.RandomHorizontalFlip(p=0.5)]
    # Lật dọc KHÔNG dùng mặc định: ảnh cỏ dại có hướng trọng lực (đất/trời);
    # lật dọc tạo mẫu phi thực tế -> chỉ thử như ablation có kiểm soát.
    if aug in ("color", "trivial", "randaug"):
        tf.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02))
    if aug == "trivial":
        tf.append(transforms.TrivialAugmentWide())
    elif aug == "randaug":
        tf.append(transforms.RandAugment(num_ops=2, magnitude=9))
    elif aug not in ("basic", "color", "trivial", "randaug"):
        raise ValueError(f"aug không hỗ trợ: {aug}")
    tf += [transforms.ToTensor(), normalize]
    return transforms.Compose(tf)


class DeepWeedsDataset(torch.utils.data.Dataset):
    """Dataset đọc ảnh theo DataFrame. __getitem__ -> (tensor, int label, filename)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path = self.images_dir / str(row["Filename"])
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["Label"]), str(row["Filename"])


def seed_worker(worker_id: int):
    w = torch.initial_seed() % 2**32
    np.random.seed(w + worker_id)
    random.seed(w + worker_id)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """train=True: shuffle (hoặc WeightedRandomSampler nếu sampler='balanced'). Val/test: giữ thứ tự."""
    ds = DeepWeedsDataset(df, images_dir, transform)
    use_sampler = None
    shuffle = train
    if sampler == "balanced":
        counts = df["Label"].value_counts().sort_index()
        w = 1.0 / counts.reindex(range(NUM_CLASSES), fill_value=1).to_numpy(dtype=np.float64)
        weights = torch.tensor([w[int(l)] for l in df["Label"].to_numpy()], dtype=torch.double)
        use_sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    elif sampler is not None:
        raise ValueError(f"sampler không hỗ trợ: {sampler}")
    g = torch.Generator()
    g.manual_seed(0)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=use_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=(train and len(df) > batch_size),
        worker_init_fn=seed_worker, generator=g if num_workers == 0 else None,
    )
