import os

import PIL.Image
import torch
import torchvision.transforms as transforms


def _get_transform():
    """Return the standard preprocessing transform for model input."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def tensor_from_pil(pil_img, use_cuda=False):
    """Convert a PIL image to a normalized model input tensor."""
    image = pil_img.convert("RGB")
    transform = _get_transform()
    image = transform(image).float()
    image = image.unsqueeze(0)

    if use_cuda:
        return image.cuda()
    return image


def image_loader(image_name, use_cuda=False):
    """Load and preprocess an image path for the cascade models."""
    image = PIL.Image.open(image_name)
    return tensor_from_pil(image, use_cuda=use_cuda)


def cascade_predict(element, CNN1, CNN2, CNN3, CNN4):
    """Run the hierarchical cascade on a preprocessed input tensor."""
    out1 = torch.sigmoid(CNN1(element))

    if out1[0] > 0.5:
        out2 = torch.sigmoid(CNN2(element))
        if out2[0] < 0.5:
            res = "Lung: Benign"
        else:
            out4 = torch.sigmoid(CNN4(element))
            if out4[0] < 0.5:
                res = "Lung: Malignant - ACA"
            else:
                res = "Lung: Malignant - SCC"
    else:
        out3 = torch.sigmoid(CNN3(element))
        if out3[0] > 0.5:
            res = "Colon: Benign"
        else:
            res = "Colon: Malignant"

    return res


def classify_image(img_path, CNN1, CNN2, CNN3, CNN4, use_cuda=False):
    """Classify a single image by its file path."""
    element = image_loader(img_path, use_cuda=use_cuda)
    return cascade_predict(element, CNN1, CNN2, CNN3, CNN4)


def classify_image_from_pil(pil_img, CNN1, CNN2, CNN3, CNN4, use_cuda=False):
    """Classify a PIL image directly (e.g., from an uploaded file)."""
    element = tensor_from_pil(pil_img, use_cuda=use_cuda)
    return cascade_predict(element, CNN1, CNN2, CNN3, CNN4)


def model_loader(net, model_path, use_cuda=False):
    """Load a state dict into a model.

    ``weights_only=False`` is used only for trusted, locally trained
    checkpoints in this repository. Do not load untrusted files this way.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    map_location = torch.device("cuda" if use_cuda else "cpu")

    try:
        state = torch.load(model_path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions may not support the weights_only argument.
        state = torch.load(model_path, map_location=map_location)

    net.load_state_dict(state)

    if use_cuda:
        net = net.cuda()

    return net
