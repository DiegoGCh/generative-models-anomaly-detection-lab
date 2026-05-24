"""
MVTec AD Dataset loader.
- Training: solo train/good/ (imágenes normales)
- Test: todas las imágenes de test + ground truth masks
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


def get_train_transforms(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),           # [0, 1]
    ])


def get_test_transforms(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),           # [0, 1]
    ])


def get_mask_transforms(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])


class MVTecTrainDataset(Dataset):
    """Solo imágenes good de train. Sin máscaras."""

    def __init__(self, root: str, category: str, img_size: int = 256):
        self.transform = get_train_transforms(img_size)
        good_dir = Path(root) / category / "train" / "good"
        self.images = sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg"))
        assert len(self.images) > 0, f"No images found in {good_dir}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img)


class MVTecTestDataset(Dataset):
    """
    Todas las imágenes de test con su máscara ground truth.
    Imágenes 'good' tienen máscara = todo ceros.
    """

    def __init__(self, root: str, category: str, img_size: int = 256):
        self.img_transform  = get_test_transforms(img_size)
        self.mask_transform = get_mask_transforms(img_size)

        test_dir = Path(root) / category / "test"
        gt_dir   = Path(root) / category / "ground_truth"

        self.samples = []   # (img_path, mask_path_or_None, label)

        for defect_type in sorted(os.listdir(test_dir)):
            defect_path = test_dir / defect_type
            if not defect_path.is_dir():
                continue

            is_good = defect_type == "good"

            for img_path in sorted(defect_path.glob("*.png")) + sorted(defect_path.glob("*.jpg")):
                if is_good:
                    mask_path = None
                else:
                    # ground_truth/<defect_type>/<name>_mask.png
                    mask_name = img_path.stem + "_mask.png"
                    mask_path = gt_dir / defect_type / mask_name

                self.samples.append((img_path, mask_path, 0 if is_good else 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        else:
            # good image → máscara todo ceros
            mask = torch.zeros(1, img.shape[1], img.shape[2])

        return img, mask, label


def get_dataloaders(root: str, category: str, img_size: int = 256, batch_size: int = 16):
    train_ds = MVTecTrainDataset(root, category, img_size)
    test_ds  = MVTecTestDataset(root, category, img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False, num_workers=0, pin_memory=True)

    print(f"[{category}] train={len(train_ds)} | test={len(test_ds)}")
    return train_loader, test_loader
