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
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
 
    def forward(self, x):
        # 三张图片合为一张图片直接卷积
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        #三张图片单独卷积最后拼到一起
        x = x.transpose(0,1)
        for category in x:
            output = self.layer1(category)
            output = self.layer2(output)
            output = self.layer3(output)
            output = self.layer4(output)
            output = self.fc_1(output)
            if 

        return x