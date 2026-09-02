import io
import os
import random
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from models import (
    CNN1_LungColon,
    CNN2_LungClassifier,
    CNN3_ColonClassifier,
    CNN4_LungMalignant,
    HistopathDetector,
)
from utilities import predict_funcs


BASE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = BASE_DIR / "checkpoint_files"
DATASET_ROOT = BASE_DIR / "dataset" / "lung_colon_image_set"
STATIC_DIR = Path(__file__).resolve().parent / "static"

CASCADE_CHECKPOINTS = {
    "cnn1": CHECKPOINT_DIR / "model_CNN1_LungColon_bs256_lr0.001_best",
    "cnn2": CHECKPOINT_DIR / "model_CNN2_LungClassifier_bs150_lr0.001_best",
    "cnn3": CHECKPOINT_DIR / "model_CNN3_ColonClassifier_bs256_lr0.001_best",
    "cnn4": CHECKPOINT_DIR / "model_CNN4_LungMalignant_bs64_lr0.001_best",
}

DETECTOR_CHECKPOINT = CHECKPOINT_DIR / "model_HistopathDetector_bs64_lr0.001_best"


def load_models():
    CNN1 = CNN1_LungColon.CNN1_LungColon()
    CNN2 = CNN2_LungClassifier.CNN2_LungClassifier()
    CNN3 = CNN3_ColonClassifier.CNN3_ColonClassifier()
    CNN4 = CNN4_LungMalignant.CNN4_LungMalignant()
    detector = HistopathDetector.HistopathDetector()

    CNN1 = predict_funcs.model_loader(CNN1, str(CASCADE_CHECKPOINTS["cnn1"]))
    CNN2 = predict_funcs.model_loader(CNN2, str(CASCADE_CHECKPOINTS["cnn2"]))
    CNN3 = predict_funcs.model_loader(CNN3, str(CASCADE_CHECKPOINTS["cnn3"]))
    CNN4 = predict_funcs.model_loader(CNN4, str(CASCADE_CHECKPOINTS["cnn4"]))
    detector = predict_funcs.model_loader(detector, str(DETECTOR_CHECKPOINT))

    return CNN1, CNN2, CNN3, CNN4, detector


def is_histopathology_image(pil_image, detector):
    element = predict_funcs.tensor_from_pil(pil_image, use_cuda=False)
    detector.eval()
    with torch.no_grad():
        logit = detector(element)
        prob = torch.sigmoid(logit).item()
    detector.train()
    return prob > 0.5, prob


def cascade_predict_with_confidence(element, CNN1, CNN2, CNN3, CNN4):
    out1 = torch.sigmoid(CNN1(element))
    if out1[0] > 0.5:
        out2 = torch.sigmoid(CNN2(element))
        if out2[0] < 0.5:
            label = "Lung: Benign"
            confidence = (1.0 - out2[0]).item()
        else:
            out4 = torch.sigmoid(CNN4(element))
            if out4[0] < 0.5:
                label = "Lung: Malignant - ACA"
                confidence = (1.0 - out4[0]).item()
            else:
                label = "Lung: Malignant - SCC"
                confidence = out4[0].item()
    else:
        out3 = torch.sigmoid(CNN3(element))
        if out3[0] > 0.5:
            label = "Colon: Benign"
            confidence = out3[0].item()
        else:
            label = "Colon: Malignant"
            confidence = (1.0 - out3[0]).item()
    return label, confidence


def image_from_upload(file: UploadFile):
    try:
        image = Image.open(io.BytesIO(file.file.read()))
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return image


def image_from_dataset_path(dataset_path: str):
    base_resolved = DATASET_ROOT.resolve()
    candidate = (DATASET_ROOT / dataset_path).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset image path.")

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Dataset image not found.")

    try:
        image = Image.open(candidate)
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return image


app = FastAPI(title="HistoScope")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def startup():
    global CNN1, CNN2, CNN3, CNN4, detector
    CNN1, CNN2, CNN3, CNN4, detector = load_models()


@app.get("/")
def read_root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/random")
def random_images():
    class_dirs = [
        DATASET_ROOT / "colon_image_sets" / "colon_aca",
        DATASET_ROOT / "colon_image_sets" / "colon_n",
        DATASET_ROOT / "lung_image_sets" / "lung_aca",
        DATASET_ROOT / "lung_image_sets" / "lung_n",
        DATASET_ROOT / "lung_image_sets" / "lung_scc",
    ]

    selected = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for class_dir in class_dirs:
        if not class_dir.exists():
            raise HTTPException(status_code=500, detail="Dataset directory missing.")
        files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        if not files:
            raise HTTPException(status_code=500, detail="No images found in class directory.")
        chosen = random.choice(files)
        relative = chosen.relative_to(DATASET_ROOT).as_posix()
        selected.append(relative)

    return JSONResponse({"images": selected})


@app.get("/dataset_image/{dataset_path:path}")
def serve_dataset_image(dataset_path: str):
    base_resolved = DATASET_ROOT.resolve()
    candidate = (DATASET_ROOT / dataset_path).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset image path.")

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Dataset image not found.")

    return FileResponse(str(candidate))


@app.post("/predict")
async def predict(
    file: UploadFile = File(None),
    dataset_path: str = Form(None),
):
    if file is None and dataset_path is None:
        raise HTTPException(status_code=400, detail="No image provided.")

    if file is not None:
        pil_image = image_from_upload(file)
    else:
        pil_image = image_from_dataset_path(dataset_path)

    # Histopathology validation
    valid, detector_prob = is_histopathology_image(pil_image, detector)

    if not valid:
        return JSONResponse({
            "valid": False,
            "message": "This does not appear to be a histopathology image. Please upload a tissue slide image.",
            "detector_confidence": round(detector_prob, 4),
        })

    # Run cascade
    element = predict_funcs.tensor_from_pil(pil_image, use_cuda=False)
    label, confidence = cascade_predict_with_confidence(element, CNN1, CNN2, CNN3, CNN4)

    return JSONResponse({
        "valid": True,
        "prediction": label,
        "confidence": round(confidence, 4),
        "detector_confidence": round(detector_prob, 4),
    })
