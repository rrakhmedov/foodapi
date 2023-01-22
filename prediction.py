import skimage.io as io
from torchvision import datasets, transforms
from torchvision import models as model
import torch
import torch.nn as nn
import numpy as np



def full_pipe(image, model):
    image_check = io.imread(image)[:,:,:3]
    transform_test = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=(224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    test = transform_test(image_check)
    test.unsqueeze(dim=0).shape



    result = model(test.unsqueeze(dim=0))
    _, result = torch.topk(result, 3)

    return result.tolist()




