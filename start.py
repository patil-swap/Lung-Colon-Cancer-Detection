import os
import argparse

from models import CNN1_LungColon, CNN2_LungClassifier, CNN3_ColonClassifier, CNN4_LungMalignant
from utilities import predict_funcs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoint_files")
DEMO_DIR = os.path.join(BASE_DIR, "dataset", "demo")

CHECKPOINT_PATHS = {
    "CNN1": os.path.join(CHECKPOINT_DIR, "model_CNN1_LungColon_bs256_lr0.001_best"),
    "CNN2": os.path.join(CHECKPOINT_DIR, "model_CNN2_LungClassifier_bs150_lr0.001_best"),
    "CNN3": os.path.join(CHECKPOINT_DIR, "model_CNN3_ColonClassifier_bs256_lr0.001_best"),
    "CNN4": os.path.join(CHECKPOINT_DIR, "model_CNN4_LungMalignant_bs64_lr0.001_best"),
}

DEMO_IMAGES = {
    1: os.path.join(DEMO_DIR, "1.jpeg"),
    2: os.path.join(DEMO_DIR, "2.jpeg"),
    3: os.path.join(DEMO_DIR, "3.jpeg"),
    4: os.path.join(DEMO_DIR, "4.jpeg"),
    5: os.path.join(DEMO_DIR, "5.jpeg"),
    6: os.path.join(DEMO_DIR, "6.jpeg"),
}

MENU_TEXT = (
    "Select one of the following options:\n"
    "1. Demo Image 1 # Labeled as Lung - Malignant - SCC\n"
    "2. Demo Image 2 # Labeled as Colon - Benign\n"
    "3. Demo Image 3 # Labeled as Colon - Malignant\n"
    "4. Demo Image 4 # Labeled as Lung - Benign\n"
    "5. Demo Image 5 # Labeled as Lung - Malignant - ACA\n"
    "6. Demo Image 6 # Labeled as Colon - Malignant - ACA\n"
)

def load_cascade():
    for model_name, path in CHECKPOINT_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{model_name} checkpoint not found: {path}")

    CNN1 = CNN1_LungColon.CNN1_LungColon()
    CNN1 = predict_funcs.model_loader(CNN1, CHECKPOINT_PATHS["CNN1"])

    CNN2 = CNN2_LungClassifier.CNN2_LungClassifier()
    CNN2 = predict_funcs.model_loader(CNN2, CHECKPOINT_PATHS["CNN2"])

    CNN3 = CNN3_ColonClassifier.CNN3_ColonClassifier()
    CNN3 = predict_funcs.model_loader(CNN3, CHECKPOINT_PATHS["CNN3"])

    CNN4 = CNN4_LungMalignant.CNN4_LungMalignant()
    CNN4 = predict_funcs.model_loader(CNN4, CHECKPOINT_PATHS["CNN4"])

    return CNN1, CNN2, CNN3, CNN4

def parse_image_selection():
    parser = argparse.ArgumentParser(
        description="Classify a lung/colon histopathology image using the pretrained 4-CNN cascade."
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Optional path to a custom image. If omitted, an interactive demo menu is shown.",
    )
    args = parser.parse_args()

    if args.image:
        image_path = os.path.abspath(args.image)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
        return image_path

    while True:
        try:
            option = int(input(MENU_TEXT))
        except (ValueError, EOFError):
            print("Invalid input. Please enter an integer between 1 and 5.")
            continue

        if option in DEMO_IMAGES:
            image_path = DEMO_IMAGES[option]
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Demo image not found: {image_path}")
            return image_path

        print("Invalid choice. Please select a number between 1 and 5.")

def main():
    CNN1, CNN2, CNN3, CNN4 = load_cascade()
    image_path = parse_image_selection()

    prediction = predict_funcs.classify_image(
        image_path,
        CNN1,
        CNN2,
        CNN3,
        CNN4,
    )

    print(f"Predicted as {prediction}")

if __name__ == "__main__":
    main()
