import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from PIL import Image
from torch import nn

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

def generate_dataloader_single(category,category_dict):
    train_fea = []
    train_label = []
    for i in range(split):
        print(i)
        mol = data[category][i]
        mol_img = Image.open('./' + category + '\\' + category_dict[mol] + '.png')
        mol_data = transform(mol_img)
        train_fea.append(np.array(mol_data))
        train_label.append(data['Output'][i])
    
    feaTrain = torch.from_numpy(np.array(train_fea))
    targetsTrain = torch.from_numpy(np.array(train_label))
    data_loader = torch.utils.data.TensorDataset(feaTrain,targetsTrain)
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size = batch_size, shuffle=True)

    return data_loader

def generate_dataloader_total():
    train_fea = []
    train_label = []
    dicts = [Catalyst,Imine,Thiol]
    for i in range(split):
        train_reaction = []
        print(i)
        for index,category in enumerate(['Catalyst','Imine','Thiol']):
            mol = data[category][i]
            mol_img = Image.open('./' + category + '\\' + dicts[index][mol] + '.png')
            mol_data = transform(mol_img)
            train_fea.append(np.array(mol_data))
            train_label.append(data['Output'][i])
    
    feaTrain = torch.from_numpy(np.array(train_fea))
    targetsTrain = torch.from_numpy(np.array(train_label))
    data_loader = torch.utils.data.TensorDataset(feaTrain,targetsTrain)
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size = batch_size, shuffle=True)

    return data_loader

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
    transforms.ToTensor(),  # ??????????????????Tensor,????????????[0,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # ????????????[-1,1]
])

data = pd.read_csv(file)

Catalyst = get_molecule_dict('Catalyst')
Imine = get_molecule_dict('Imine')
Thiol = get_molecule_dict('Thiol')



Catalyst_loader = generate_dataloader('Catalyst',Catalyst)
Imine_loader = generate_dataloader('Imine',Imine)
Thiol_loader = generate_dataloader('Thiol',Thiol)

model = CNN()
model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    for batch_idx,(cat_fea,label) in enumerate(Catalyst_loader):
        cat_fea = cat_fea.to(DEVICE)
        output_cat = model(cat_fea)

        print(output_cat.size())

    
    if epoch == 0:
        break

