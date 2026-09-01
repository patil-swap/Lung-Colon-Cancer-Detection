import os
import random

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_files(directory):
    """Return a sorted list of image file paths in a class directory."""
    files = []
    for fname in sorted(os.listdir(directory)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(os.path.join(directory, fname))
    return files


def get_train_transforms():
    """Mild augmentation transforms for training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_eval_transforms():
    """Inference/validation transforms without augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class BinaryDataset(Dataset):
    """Dataset that maps selected source class folders to binary labels.

    Args:
        samples: List of ``(image_path, source_class_name)`` tuples.
        class_to_label: Mapping from source class name to binary label.
        transform: Optional torchvision transform.
    """

    def __init__(self, samples, class_to_label, transform=None):
        self.samples = samples
        self.class_to_label = class_to_label
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, source_class = self.samples[idx]
        label = self.class_to_label[source_class]

        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_class_dirs(dataset_root):
    """Map the five source classes to their actual directories in this repo."""
    return {
        "colon_aca": os.path.join(dataset_root, "colon_image_sets", "colon_aca"),
        "colon_n": os.path.join(dataset_root, "colon_image_sets", "colon_n"),
        "lung_aca": os.path.join(dataset_root, "lung_image_sets", "lung_aca"),
        "lung_n": os.path.join(dataset_root, "lung_image_sets", "lung_n"),
        "lung_scc": os.path.join(dataset_root, "lung_image_sets", "lung_scc"),
    }


def collect_samples(class_dirs, class_to_label):
    """Collect image paths for every source class used by a binary task."""
    samples = []
    for class_name in sorted(class_to_label.keys()):
        if class_name not in class_dirs:
            raise FileNotFoundError(
                f"Missing class directory mapping for: {class_name}"
            )

        class_dir = class_dirs[class_name]
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        for path in list_image_files(class_dir):
            samples.append((path, class_name))

    samples.sort(key=lambda item: item[0])
    return samples


def create_binary_loaders(
    dataset_root,
    class_to_label,
    batch_size,
    val_split=0.15,
    num_workers=2,
    seed=42,
):
    """Create train/val loaders for one binary cascade task.

    Images are loaded directly from the raw ``lung_colon_image_set``
    directory layout and split deterministically with ``random_split``.

    Returns:
        train_loader, val_loader, train_subset, val_subset, pos_weight
    """
    class_dirs = build_class_dirs(dataset_root)
    samples = collect_samples(class_dirs, class_to_label)

    total = len(samples)
    if total == 0:
        raise RuntimeError("No samples found for the requested task.")

    val_count = int(total * val_split)
    indices = list(range(total))
    random.Random(seed).shuffle(indices)

    train_indices = indices[val_count:]
    val_indices = indices[:val_count]

    full_train_dataset = BinaryDataset(
        samples,
        class_to_label,
        transform=get_train_transforms(),
    )
    full_val_dataset = BinaryDataset(
        samples,
        class_to_label,
        transform=get_eval_transforms(),
    )

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    train_labels = [
        class_to_label[samples[i][1]]
        for i in train_indices
    ]
    counts = torch.bincount(
        torch.tensor(train_labels, dtype=torch.long),
        minlength=2,
    ).float()
    pos_weight = float(counts[0] / counts[1].clamp(min=1))

    return train_loader, val_loader, train_subset, val_subset, pos_weight


# Binary class mappings matching the cascade decision logic in predict_funcs.py.
CNN1_CLASS_TO_LABEL = {
    "colon_aca": 0,
    "colon_n": 0,
    "lung_aca": 1,
    "lung_n": 1,
    "lung_scc": 1,
}

CNN2_CLASS_TO_LABEL = {
    "lung_n": 0,      # benign
    "lung_aca": 1,    # malignant
    "lung_scc": 1,    # malignant
}

CNN3_CLASS_TO_LABEL = {
    "colon_aca": 0,   # malignant
    "colon_n": 1,     # benign
}

CNN4_CLASS_TO_LABEL = {
    "lung_aca": 0,    # ACA
    "lung_scc": 1,    # SCC
}


DEFAULT_BATCH_SIZES = {
    "cnn1": 256,
    "cnn2": 150,
    "cnn3": 256,
    "cnn4": 64,
}


def get_cascade_loaders(
    dataset_root="dataset/lung_colon_image_set",
    batch_sizes=None,
    val_split=0.15,
    num_workers=2,
    seed=42,
):
    """Create loaders for all four binary cascade models.

    Args:
        dataset_root: Path to ``lung_colon_image_set``.
        batch_sizes: Optional dict overriding default task batch sizes.
        val_split: Fraction of task data reserved for validation.
        num_workers: DataLoader worker count.
        seed: Random seed for deterministic train/val splits.

    Returns:
        Dictionary keyed by ``"cnn1"`` through ``"cnn4"``. Each value
        contains ``train``, ``val``, ``train_dataset``, ``val_dataset``,
        and ``pos_weight``.
    """
    batch_sizes = batch_sizes or DEFAULT_BATCH_SIZES

    task_maps = {
        "cnn1": CNN1_CLASS_TO_LABEL,
        "cnn2": CNN2_CLASS_TO_LABEL,
        "cnn3": CNN3_CLASS_TO_LABEL,
        "cnn4": CNN4_CLASS_TO_LABEL,
    }

    loaders = {}

    for task_name, class_to_label in task_maps.items():
        train_loader, val_loader, train_dataset, val_dataset, pos_weight = create_binary_loaders(
            dataset_root=dataset_root,
            class_to_label=class_to_label,
            batch_size=batch_sizes[task_name],
            val_split=val_split,
            num_workers=num_workers,
            seed=seed,
        )

        loaders[task_name] = {
            "train": train_loader,
            "val": val_loader,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "pos_weight": pos_weight,
        }

    return loaders
