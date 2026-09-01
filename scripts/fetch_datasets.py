import os
import shutil
import zipfile
from pathlib import Path

from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
HISTOPATH_DIR = DATASET_DIR / "lung_colon_image_set"
NON_HISTOPATH_DIR = DATASET_DIR / "non_histopath" / "caltech101"
KAGGLE_DATASET = "andrewmvd/lung-and-colon-cancer-histopathological-images"


def find_histopath_source(extract_dir: Path) -> Path:
    """Locate the directory containing colon_image_sets and lung_image_sets."""
    for candidate in extract_dir.rglob("colon_image_sets"):
        if (candidate.parent / "lung_image_sets").exists():
            return candidate.parent
    raise FileNotFoundError(
        "Could not find histopathology source directory with "
        "colon_image_sets and lung_image_sets."
    )


def fetch_histopath_dataset():
    """Download and prepare the Kaggle histopathology dataset."""
    if HISTOPATH_DIR.exists():
        print(f"Histopathology dataset already exists at {HISTOPATH_DIR}")
        return

    print("Downloading Kaggle dataset...")
    try:
        import kaggle
    except ImportError:
        raise ImportError(
            "Kaggle API package is required. Install with: pip install kaggle"
        )

    # Kaggle API will download to current working directory by default.
    # Use a temporary directory for extraction.
    extract_root = DATASET_DIR / "_kaggle_temp"
    extract_root.mkdir(parents=True, exist_ok=True)

    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(extract_root),
        unzip=False,
    )

    # Find downloaded zip file
    zip_files = list(extract_root.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError("No downloaded zip file found.")

    zip_path = zip_files[0]
    print(f"Extracting {zip_path.name}...")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_root)

    # Locate source directory and move to final location
    source_dir = find_histopath_source(extract_root)
    print(f"Found source directory: {source_dir}")

    shutil.move(str(source_dir), str(HISTOPATH_DIR))
    shutil.rmtree(extract_root, ignore_errors=True)

    print(f"Histopathology dataset ready at {HISTOPATH_DIR}")


def fetch_non_histopath_dataset():
    """Download Caltech101 using torchvision."""
    if NON_HISTOPATH_DIR.exists():
        print(f"Non-histopathology dataset already exists at {NON_HISTOPATH_DIR}")
        return

    print("Downloading Caltech101 dataset...")
    NON_HISTOPATH_DIR.parent.mkdir(parents=True, exist_ok=True)

    datasets.Caltech101(
        root=str(NON_HISTOPATH_DIR.parent),
        download=True,
    )

    print(f"Non-histopathology dataset ready at {NON_HISTOPATH_DIR}")


def main():
    fetch_histopath_dataset()
    fetch_non_histopath_dataset()
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
