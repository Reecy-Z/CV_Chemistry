import torch
import numpy as np
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size=3),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True)
        )
 
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
 
        self.layer3 = nn.Sequential(
            nn.Conv2d(5, 3, kernel_size=3),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
 
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
 
        self.fc_1 = nn.Sequential(
            nn.Linear(15987, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128)
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(15987, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128)
        )
 
    def forward(self, x):
        # 三张图片合为一张图片直接卷积
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        return x

model = CNN()

a = np.arange(2160000).reshape(2,4,3,300,300)
a = torch.from_numpy(a)
a = a.transpose(0,1)
for index,i in enumerate(a):
    output = model(i.to(torch.float))
    if index == 0:
        output_cat = output
    else:
        output_cat = torch.cat((output_cat,output),dim = 1)
    print(output_cat.size())
# torch.Size([2, 4, 3, 300, 300])
# model = CNN()
# output = model(a.to(torch.float))
# print(output.size())
