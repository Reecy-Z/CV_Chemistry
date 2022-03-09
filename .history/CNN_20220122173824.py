import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from PIL import Image
from torch import nn
from xgboost import train

def squared_loss(y_true, y_pred):
    """Compute the squared loss for regression.
    """
    return ((y_true - y_pred) ** 2).mean() / 2

def get_molecule_dict(category):
    category_dict = {}
    i = 1
    for mol in data[category]:
        if mol not in category_dict.keys():
            category_dict[mol] = category + '_' +str(i)
            i += 1
    return category_dict

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True)
        )
 
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
 
        self.layer3 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True)
        )
 
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
 
        self.fc = nn.Sequential(
            nn.Linear(106580, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 1)
        )
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

file = '19_science_total.csv'
split = 600
learning_rate = 0.002
epochs = 1000
batch_size = 1
DEVICE = torch.device("cuda:0")

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
])

data = pd.read_csv(file)

Catalyst = get_molecule_dict('Catalyst')
Imine = get_molecule_dict('Imine')
Thiol = get_molecule_dict('Thiol')

train_catalyst_fea = []
train_imine_fea = []
train_thiol_fea = []
train_label = []
for i in range(split):
    print(i)

    catalyst = data['Catalyst'][i]
    imine = data['Imine'][i]
    thiol = data['Thiol'][i]

    catalyst_img = Image.open('./Catalyst\\' + Catalyst[catalyst] + '.png')
    imine_img = Image.open('./Imine\\' + Imine[imine] + '.png')
    thiol_img = Image.open('./Thiol\\' + Thiol[thiol] + '.png')

    catalyst_data = transform(catalyst_img)
    imine_data = transform(imine_img)
    thiol_data = transform(thiol_img)

    train_catalyst_fea.append(np.array(catalyst_data))
    train_imine_fea.append(np.array(imine_data))
    train_thiol_fea.append(np.array(thiol_data))

    train_label.append(data['Output'][i])

test_catalyst_fea = []
test_imine_fea = []
test_thiol_fea = []
test_label = []
for i in range(split,1075):
    print(i)
    test_reaction = []

    catalyst = data['Catalyst'][i]
    imine = data['Imine'][i]
    thiol = data['Thiol'][i]

    catalyst_img = Image.open('./Catalyst\\' + Catalyst[catalyst] + '.png')
    imine_img = Image.open('./Imine\\' + Imine[imine] + '.png')
    thiol_img = Image.open('./Thiol\\' + Thiol[thiol] + '.png')

    catalyst_data = transform(catalyst_img)
    imine_data = transform(imine_img)
    thiol_data = transform(thiol_img)

    test_catalyst_fea.append(np.array(catalyst_data))
    test_imine_fea.append(np.array(imine_data))
    test_thiol_fea.append(np.array(thiol_data))

    test_label.append(data['Output'][i])

Train_catalyst_fea = torch.from_numpy(np.array(train_catalyst_fea))
Train_imine_fea = torch.from_numpy(np.array(train_imine_fea))
Train_thiol_fea = torch.from_numpy(np.array(train_thiol_fea))

targetsTrain = torch.from_numpy(np.array(train_label))

Catalyst_loader = torch.utils.data.TensorDataset(Train_catalyst_fea,targetsTrain)
Catalyst_loader = torch.utils.data.DataLoader(Catalyst_loader, batch_size = batch_size, shuffle=True)

def generate_dataloader(category):
    train_fea = []
    train_label = []
    for i in range(split):
        print(i)

        catalyst = data[category][i]
        imine = data['Imine'][i]
        thiol = data['Thiol'][i]

        catalyst_img = Image.open('./Catalyst\\' + Catalyst[catalyst] + '.png')
        imine_img = Image.open('./Imine\\' + Imine[imine] + '.png')
        thiol_img = Image.open('./Thiol\\' + Thiol[thiol] + '.png')

        catalyst_data = transform(catalyst_img)
        imine_data = transform(imine_img)
        thiol_data = transform(thiol_img)

        train_catalyst_fea.append(np.array(catalyst_data))
        train_imine_fea.append(np.array(imine_data))
        train_thiol_fea.append(np.array(thiol_data))

        train_label.append(data['Output'][i])


model = CNN()
model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    for cat in Train_catalyst_fea:
        cat = cat.to(DEVICE)
        output = model(cat)
        # output = [output[0],output[1],output[2]]
        # output = torch.cat(output,1)
        print(output.size())
    
    if epoch == 0:
        break

