from convCal import effNet
import torch
from  dataLoader import get_data
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np
import torchvision
import torch
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def matGap(mat):
    mat[60,:] *= 0
    mat[188,:] *= 100
    mat[200,:] *= 100
    mat[207,:] *= 100
    mat[206,:] *= 100
    return mat

if __name__ == "__main__":
    train_data, val_data = get_data(load=False,batch_size=8)
    model = effNet()
    model.load_state_dict(torch.load('bestmodel.pth'), strict=True)
    model = model.to(device)
    target_layer = [model.backbone.features[-1]] 

    # 4.初始化GradCAM，包括模型，目标层以及是否使用cuda
    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)

    # 5.选定目标类别，如果不设置，则默认为分数最高的那一类
    targets = None 
    # targets = [ClassifierOutputTarget(2)] 

    for batch, (input_data, label) in enumerate(train_data):
        pred = model(input_data)
        # 6. 计算cam
        grayscale_cam = cam(input_tensor=input_data, targets=targets)
            
        gray = grayscale_cam[0,:]
        norm_img = np.zeros(gray.shape)

        norm_img=cv2.normalize(gray , norm_img, 0, 255, cv2.NORM_MINMAX)
        norm_img = np.asarray(norm_img, dtype=np.uint8)

        heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET) # 注意此处的三通道热力图是cv2专有的GBR排列
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)# 将BGR图像转为RGB图像

        # grayscale_cam = matGap(grayscale_cam[0,:]*256)
        
        cv2.imwrite('2.jpg',heat_img)
        a =1
        # 加上 aug_smooth=True(应用水平翻转组合，并通过[1.0，1.1，0.9]对图像进行多路复用，使CAM围绕对象居中)
        # eigen_smooth=True(去除大量噪声)
