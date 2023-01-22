from fastapi import FastAPI, Form, Request
from typing import Union
import uvicorn
import pandas as pd

import torch
from torchvision import datasets, transforms
import skimage.io as io
import torch.nn as nn
from torchvision import models as model
from prediction import full_pipe
from fastapi import File
from fastapi import UploadFile
from fastapi.responses import FileResponse


resnet = model.resnet50(pretrained=False)
resnet.fc = nn.Linear(2048,251)
resnet.load_state_dict(torch.load('model_torchfile', map_location=torch.device('cpu')))
resnet.eval()

transform_test = transforms.Compose([    transforms.ToPILImage(),
                                transforms.Resize(size=(224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])

app = FastAPI()

df_tr = pd.read_csv(r'output.csv',sep=';')

@app.get('/index')
def hello_world():
    print('Hello world')
    return {"code":"success"}

@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):
    image = full_pipe(file.file, resnet)
    print(image)
    # print( df_tr[df_tr['productId'] ==image[0][0]])
    return df_tr[df_tr['predictid'].isin(image[0])].iloc[:,:5].to_dict(orient="records")



if __name__ == "__main__":
    uvicorn.run(app, port = 9314, host = '0.0.0.0')