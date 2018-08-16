import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channels,stride=1):
        super(BasicBlock,self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channel,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels))
        self.shortCut = nn.Sequential()
        if stride != 1 or in_channel != out_channels:
            self.shortCut = nn.Sequential(
                nn.Conv2d(in_channel,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels))
        if out_channels == 64:
            self.avgpool2d = nn.AvgPool2d(32,stride=1)
        elif out_channels == 128:
            self.avgpool2d = nn.AvgPool2d(16,stride=1) 
        elif out_channels == 256:
            self.avgpool2d = nn.AvgPool2d(8,stride=1)
        elif out_channels == 512:
            self.avgpool2d = nn.AvgPool2d(4,stride=1)
        self.fc1 = nn.Linear(out_channels,out_channels//16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(out_channels//16,out_channels)
        self.sigmoid =nn.Sigmoid()

    def forward(self,input):
        residual = self.residual(input)
        shortCut = self.shortCut(input)
        se = self.avgpool2d(residual)
        se = se.view(se.size(0),-1)
        se = self.fc1(se)
        se = self.relu1(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        se = se.view(se.size(0),se.size(1),1,1)
        out = se*residual
        out += shortCut
        out = nn.functional.relu(out)
        return out

class RESNet(nn.Module):
    def __init__(self,BasicBlock):
        super(RESNet,self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer1 = self.make_layer(BasicBlock,64,2,stride=1)
        self.layer2 = self.make_layer(BasicBlock,128,2,stride=2)
        self.layer3 = self.make_layer(BasicBlock,256,2,stride=2)
        self.layer4 = self.make_layer(BasicBlock,512,2,stride=2)
        self.Dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512,100)
    def make_layer(self,block,channels,num_blocks,stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel,channels,stride))
            self.in_channel = channels
        return nn.Sequential(*layers)
    def forward(self,input):
        output = self.conv1(input)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = nn.functional.avg_pool2d(output,4)
        output = output.view(output.size(0),-1)
        output = self.fc(self.Dropout(output))
        return output
def Se_ResNet():
    return RESNet(BasicBlock)