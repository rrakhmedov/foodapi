import skimage.io as io
from torchvision import datasets, transforms
from torchvision import models as model
import torch
import torch.nn as nn
import numpy as np
def full_pipe(image):
    image_check = io.imread(image)
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
        torch.load(r'C:\Users\ramil\Desktop\Python_project1\image classification\model_20220920_080539_9'))
    resnet.eval()

    result = resnet(test.unsqueeze(dim=0))
    _, result = torch.topk(result, 1)

    return result.tolist()




