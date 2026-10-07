import argparse

import torch

from models import CNN1_LungColon, CNN2_LungClassifier, CNN3_ColonClassifier, CNN4_LungMalignant
from utilities import data_utils, train_net


def build_model(task):
    if task == "cnn1":
        return CNN1_LungColon.CNN1_LungColon()
    if task == "cnn2":
        return CNN2_LungClassifier.CNN2_LungClassifier()
    if task == "cnn3":
        return CNN3_ColonClassifier.CNN3_ColonClassifier()
    if task == "cnn4":
        return CNN4_LungMalignant.CNN4_LungMalignant()
    raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the 4-CNN cascade using improved loaders and optimizer."
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset/lung_colon_image_set",
        help="Path to the raw lung_colon_image_set directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoint_files",
        help="Directory where checkpoints and CSV metrics are saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum epochs per cascade model.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["cnn1", "cnn2", "cnn3", "cnn4"],
        default=["cnn1", "cnn2", "cnn3", "cnn4"],
        help="Which cascade models to train. Default: all four.",
    )
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA not available. Training on CPU.")

    loaders = data_utils.get_cascade_loaders(
        dataset_root=args.dataset_root,
        num_workers=args.num_workers,
    )

    learning_rates = {
        "cnn1": 0.001,
        "cnn2": 0.001,
        "cnn3": 0.001,
        "cnn4": 0.001,
    }

    batch_sizes = data_utils.DEFAULT_BATCH_SIZES

    for task in args.tasks:
        print(f"\n=== Training {task.upper()} ===")

        model = build_model(task)

        train_net.train_net(
            net=model,
            train_loader=loaders[task]["train"],
            val_loader=loaders[task]["val"],
            batch_size=batch_sizes[task],
            learning_rate=learning_rates[task],
            num_epochs=args.epochs,
            use_cuda=use_cuda,
            checkpoint_dir=args.checkpoint_dir,
            optimizer_name="adamw",
            weight_decay=1e-4,
            pos_weight=loaders[task]["pos_weight"],
            early_stopping_patience=5,
        )


if __name__ == "__main__":
    main()
