import PIL.Image
import numpy as np
import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import *
from torchvision import datasets, models, transforms
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import *
from torch.autograd import Variable

import models.CNN2_LungClassifier

def classify_image(img_path, CNN1, CNN2, CNN3, CNN4, use_cuda=False):
    element = image_loader(img_path)
    res = ""
    if use_cuda:
        out1 = torch.sigmoid(CNN1(element.cuda()))
    else:
        out1 = torch.sigmoid(CNN1(element))
    if out1[0] > 0.5:
        if use_cuda:
            out2 = torch.sigmoid(CNN2(element.cuda()))
        else:
            out2 = torch.sigmoid(CNN2(element))
        if out2[0] < 0.5:
            res = "Lung: Benign"
        else:
            if use_cuda:
                out4 = torch.sigmoid(CNN4(element.cuda()))
            else:
                out4 = torch.sigmoid(CNN4(element))
            if out4[0] < 0.5:
                res = "Lung: Malignant - ACA"
            else:
                res = "Lung: Malignant - SCC"
    else:
        if use_cuda:
            out3 = torch.sigmoid(CNN3(element.cuda()))
        else:
            out3 = torch.sigmoid(CNN3(element))
        if out3[0] > 0.5:
            res = "Colon: Benign"
        else:
            res = "Colon: Malignant"
    return res


def image_loader(image_name, use_cuda=False):
    """
    Load image and prepares it for model.
    
    Args:
        image_name: string representing path of image location
    Returns:
        image.cuda: CUDA tensor
    """
    image = PIL.Image.open(image_name)
    data_transform2 = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = data_transform2(image).float()
    image = image.unsqueeze(0)
    image = Variable(image)
    if use_cuda:
        return image.cuda()
    else:
        return image

def model_loader(net, model_path, use_cuda=False):
    if not use_cuda:
        state = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        state = torch.load(model_path)
    net.load_state_dict(state)
    if use_cuda:
        net = net.cuda()
    return net
