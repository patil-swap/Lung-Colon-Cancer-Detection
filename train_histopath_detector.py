import argparse
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models import HistopathDetector
from utilities import data_utils, train_net


PROJECT_ROOT = Path(__file__).resolve().parent
HISTOPATH_DIR = PROJECT_ROOT / "dataset" / "lung_colon_image_set"
NON_HISTOPATH_DIR = PROJECT_ROOT / "dataset" / "non_histopath" / "caltech101"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoint_files"


class BinaryImageDataset(Dataset):
    """Combine positive and negative image paths into a binary dataset."""

    def __init__(self, positive_paths, negative_paths, transform=None):
        self.samples = []
        for p in positive_paths:
            self.samples.append((p, 1))
        for p in negative_paths:
            self.samples.append((p, 0))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def collect_histopath_images(limit=None):
    """Collect image paths from the histopathology dataset."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = []
    for class_dir in HISTOPATH_DIR.rglob("*"):
        if class_dir.is_dir():
            for file in class_dir.iterdir():
                if file.suffix.lower() in extensions:
                    paths.append(file)
    if limit is not None:
        random.shuffle(paths)
        paths = paths[:limit]
    return paths

def collect_caltech_images(limit=None):
    """Collect image paths directly from the extracted Caltech101 folder.

    The downloaded Caltech101 dataset is expected to have images under:

        dataset/non_histopath/caltech101/imgs/...

    We bypass torchvision.datasets.Caltech101 because its wrapper expects a
    processed layout that may not match the extracted download.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = []

    for root, _, files in os.walk(NON_HISTOPATH_DIR):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                paths.append(Path(root) / file)

    if limit is not None:
        random.shuffle(paths)
        paths = paths[:limit]

    return paths
def main():
    parser = argparse.ArgumentParser(
        description="Train HistoScope histopathology detector."
    )
    parser.add_argument(
        "--positive-limit",
        type=int,
        default=5000,
        help="Maximum number of histopathology images to use.",
    )
    parser.add_argument(
        "--negative-limit",
        type=int,
        default=5000,
        help="Maximum number of non-histopathology images to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of data for validation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if not HISTOPATH_DIR.exists():
        raise FileNotFoundError(
            f"Histopathology dataset missing at {HISTOPATH_DIR}. "
            "Run scripts/fetch_datasets.py first."
        )

    print("Collecting histopathology images...")
    positive_paths = collect_histopath_images(limit=args.positive_limit)
    print(f"Positive images: {len(positive_paths)}")

    print("Collecting Caltech101 negative images...")
    negative_paths = collect_caltech_images(limit=args.negative_limit)
    print(f"Negative images: {len(negative_paths)}")

    full_dataset = BinaryImageDataset(
        positive_paths,
        negative_paths,
        transform=data_utils.get_train_transforms(),
    )

    total = len(full_dataset)
    val_count = int(total * args.val_split)
    indices = list(range(total))
    random.shuffle(indices)
    train_indices = indices[val_count:]
    val_indices = indices[:val_count]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # Override transform for validation: no augmentation
    val_dataset.dataset.transform = data_utils.get_eval_transforms()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Calculate pos_weight: negatives / positives
    train_labels = [full_dataset.samples[i][1] for i in train_indices]
    counts = torch.bincount(
        torch.tensor(train_labels, dtype=torch.long),
        minlength=2,
    ).float()
    pos_weight = float(counts[0] / counts[1].clamp(min=1))
    print(f"pos_weight: {pos_weight:.4f}")

    model = HistopathDetector.HistopathDetector()

    use_cuda = False #torch.cuda.is_available()
    print(f"Training on {'GPU' if use_cuda else 'CPU'}...")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_net.train_net(
        net=model,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        use_cuda=use_cuda,
        checkpoint_dir=str(CHECKPOINT_DIR),
        optimizer_name="adamw",
        weight_decay=1e-4,
        pos_weight=pos_weight,
        early_stopping_patience=5,
    )


if __name__ == "__main__":
    main()
