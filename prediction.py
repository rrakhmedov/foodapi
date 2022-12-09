import skimage.io as io
from torchvision import datasets, transforms
from torchvision import models as model
import torch
import torch.nn as nn
import numpy as np
def full_pipe(image):
    image_check = io.imread(image)[:,:,:3]
    transform_test = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=(224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    test = transform_test(image_check)
    test.unsqueeze(dim=0).shape

    resnet = model.resnet50(pretrained=False)
    resnet.fc = nn.Linear(2048, 251)
    resnet.load_state_dict(
        torch.load('model_torchfile', map_location=torch.device('cpu')))
    resnet.eval()

    result = resnet(test.unsqueeze(dim=0))
    _, result = torch.topk(result, 1)

    return result.tolist()




