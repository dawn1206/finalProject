import torch.nn as nn
import torch
import numpy as np
import math

import torchvision

torch.manual_seed(0)
np.random.seed(0)
calculate_loss_over_all_values = False

input_window = 100
output_window = 5
batch_size = 16  # batch size
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# .
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term1 = torch.exp(torch.arange(0, d_model-1, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * (div_term1)  )
        pe = pe.unsqueeze(0).transpose(0, 1).to(device)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransAm(nn.Module):
    def __init__(self, feature_size=213, num_layers=2, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.linear = nn.Linear(368, 2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size,dim_feedforward=1024, nhead=5, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.conv1 = nn.Conv2d(1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.dense = nn.Linear(2961,2)
        self.ba1 = nn.BatchNorm2d(16)
        self.ba2 = nn.BatchNorm2d(32)
        self.ba3 = nn.BatchNorm2d(64)
        self.ba4 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((2, 2), 2)
        self.maxpool2 = nn.MaxPool2d((6, 6), 4)
        self.maxpool3 = nn.MaxPool2d((1,2),2)
        self.drop = nn.Dropout()
        self.softmax = nn.Softmax()
        # self.maxPool = nn.MaxPool2d((1,2))


    def transForward(self,src0):
        # src = self.maxPool(src)
        # src0 = self.maxpool3(src0)
        src = self.linear(src0)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = torch.transpose(src, 1, 2)
        src = self.pos_encoder(src)
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        return output

    def convForward(self,src):
        h1 = self.conv1(src)
        b1 = self.ba1(h1)
        b1 = self.relu(b1)
        m1 = self.maxpool1(b1)
        m1 = self.drop(m1)

        h2 = self.conv2(m1)
        b2 = self.ba2(h2)
        b2 = self.relu(b2)
        m2 = self.maxpool1(b2)
        m2 = self.drop(m2)

        h3 = self.conv3(m2)
        b3 = self.ba3(h3)
        b3 = self.relu(b3)
        m3 = self.maxpool1(b3)
        m3 = self.drop(m3)

        h4 = self.conv4(m3)
        b4 = self.ba4(h4)
        b4 = self.relu(b4)
        m4 = self.maxpool2(b4)
        m4 = self.drop(m4)
        return m4

    def forward(self, src):
        # tranSrc = torch.from_numpy(np.ndarray(1,213,266))
        tranSrc = np.ndarray((1,213,368))
        for i in src:
            tranSrc = np.concatenate((tranSrc,i),axis = 0)
        tranSrc = torch.from_numpy(tranSrc[1:])
        tranSrc = tranSrc.to(torch.float32)
        src = src.to(device)


        transOut = self.transForward(tranSrc.to(device))
        ConvOut = self.convForward(src)
        ConvOut =  ConvOut.reshape((src.shape[0],2,2816))

        out = torch.cat((transOut,ConvOut),dim = 2)
        out = self.dense(out).reshape((src.shape[0],4))
        return out

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class resNet(nn.Module):
    def __init__(self):
        super(resNet,self).__init__()
        # self.backbone = torchvision.models.efficientnet.efficientnet_b0(pretrained=True)
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone.fc = nn.Linear(in_features=512, out_features=4, bias=True)
        # self.backbone.classifier.add_module("dense",nn.Linear(1000,4))
        # self.backbone.features[0][0] = nn.Conv2d(1,32,kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # self.backbone.classifier.C
        # print(self.backbone)

    def forward(self, src):
        out = self.backbone(src.to(device))
        return out

class effNet(nn.Module):
    def __init__(self):
        super(effNet,self).__init__()
        
        self.backbone = torchvision.models.efficientnet.efficientnet_b0(pretrained=True)
        self.backbone.classifier[1] = nn.Linear(in_features=1280, out_features=4, bias=True)
        self.backbone.features[0][0] = nn.Conv2d(1,32,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        print(self.backbone)

    def forward(self, src):
        out = self.backbone(src.to(device))
        return out

class Alex(nn.Module):
    def __init__(self):
        super(Alex,self).__init__()
        self.backbone = torchvision.models.AlexNet(num_classes=4)
        self.backbone.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))

    def forward(self, src):
        out = self.backbone(src.to(device))
        return out

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Trans(nn.Module):
    def __init__(self, feature_size=220, num_layers=10, dropout=0.1):
        super(Trans, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size,dim_feedforward=1024, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(feature_size*399, 4)
        self.mlp = Mlp(feature_size*399,hidden_features=feature_size*399*4,out_features=4,drop=0.1)
    #     self.init_weights()

    # def init_weights(self):
    #     initrange = 0.1
    #     self.head.bias.data.zero_()
    #     self.head.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        tranSrc = np.ndarray((1,220,399))
        for i in src:
            n = np.ndarray((1,220,399))
            n[0][:213,:] = i
            tranSrc = np.concatenate((tranSrc,n),axis = 0)
        tranSrc = torch.from_numpy(tranSrc[1:])
        src = tranSrc.to(torch.float32)
        src = src.to(device)
        src = torch.transpose(src, 1, 2)
        src = self.pos_encoder(src.to(device))

        output = self.transformer_encoder(src)#, self.src_mask)
        output = output.view(output.size(0), -1)
        x = self.mlp(output)
        return x

class vit(nn.Module):
    def __init__(self,image_size=368,patch_size=4,num_classes=4) -> None:
        super(vit,self).__init__()
        self.backbone = torchvision.models.VisionTransformer(image_size=image_size,patch_size=patch_size,num_classes=num_classes,num_heads=10,num_layers=6,hidden_dim=2000,mlp_dim=2048)
        self.backbone.conv_proj = nn.Conv2d(1, 2000, kernel_size=(4, 4), stride=(4, 4))
        # print(self.backbone)

    def forward(self,src):
        # print(src.shape)
        conv = np.zeros((src.shape[0],1,368,368))
        conv[:,:,:src.shape[2],:368] = src[:,:,:,:368]
        # print(conv.shape)
        conv = torch.from_numpy(conv).to(torch.float32)
        out = self.backbone(conv.to(device))
        return out

class swinT(nn.Module):
    def __init__(self,image_size=264,patch_size=33,num_classes=4) -> None:
        super(swinT,self).__init__()
        self.backbone = torchvision.models.SwinTransformer(patch_size=[4,4],num_classes=num_classes,num_heads=[3, 6, 12, 24],depths=[2, 2, 6, 2],embed_dim=128,window_size=[7, 7])
    
        self.backbone.features[0] = nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        # print(self.backbone)
        # self.backbone.conv_proj = nn.Conv2d(1, 2000, kernel_size=(33, 33), stride=(33, 33))
        # self.padding = nn.ReflectionPad2d((-1,1,))
        # torchvision.models.SwinTransformer()
        # print(self.backbone)

    def forward(self,src):
        # print(src.shape)
        # conv = np.zeros((src.shape[0],1,264,264))
        # conv[:,:,:src.shape[2],:264] = src[:,:,:,:264]
        # conv = torch.from_numpy(conv).to(torch.float32)
        out = self.backbone(src.to(device))
        return out

def testConv():
    x = torch.ones(5,1,213,266)
    model = resNet()
    model.train()
    a = model(x)
    return a

def testTrans():
    x = torch.ones(5,1,213,266)
    model = TransAm()
    model.train()
    a = model(x)
    return a

if __name__ == "__main__":
    testTrans()
    # testConv()
    # res = vit()
    a=1
    
