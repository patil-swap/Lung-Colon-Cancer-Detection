import os

import PIL.Image
import torch
import torchvision.transforms as transforms


def classify_image(img_path, CNN1, CNN2, CNN3, CNN4, use_cuda=False):
    element = image_loader(img_path, use_cuda=use_cuda)

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


def image_loader(image_name, use_cuda=False):
    """Load and preprocess an image for the cascade models."""
    image = PIL.Image.open(image_name).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = transform(image).float()
    image = image.unsqueeze(0)

    if use_cuda:
        return image.cuda()
    return image


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
