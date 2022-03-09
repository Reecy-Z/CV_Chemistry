import numpy as np
import pandas as pd
import torch
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from PIL import Image
from torch import nn
from sklearn.metrics import mean_absolute_error

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
            train_reaction.append(np.array(mol_data))
        train_label.append(data['Output'][i])
        train_fea.append(train_reaction)
    
    feaTrain = torch.from_numpy(np.array(train_fea))
    targetsTrain = torch.from_numpy(np.array(train_label))
    data_loader = torch.utils.data.TensorDataset(feaTrain,targetsTrain)
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size = batch_size, shuffle=True)

    return data_loader

def generate_traindataloader_cat():
    train_fea = []
    train_label = []
    dicts = [Catalyst,Imine,Thiol]
    for i in range(split):
        train_reaction = np.empty((3,300,300))
        print(i)
        for index,category in enumerate(['Catalyst','Imine','Thiol']):
            mol = data[category][i]
            mol_img = Image.open('./' + category + '\\' + dicts[index][mol] + '.png')
            mol_data = np.array(transform(mol_img))
            train_reaction = np.concatenate([train_reaction,mol_data],1)
        train_label.append(data['Output'][i])
        train_fea.append(train_reaction)
    
    feaTrain = torch.from_numpy(np.array(train_fea))
    targetsTrain = torch.from_numpy(np.array(train_label).flatten())
    data_loader = torch.utils.data.TensorDataset(feaTrain,targetsTrain)
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size = batch_size, shuffle=True)

    return data_loader

def generate_testdataloader_cat():
    test_fea = []
    test_label = []
    dicts = [Catalyst,Imine,Thiol]
    for i in range(split,1075):
        test_reaction = np.empty((3,300,300))
        print(i)
        for index,category in enumerate(['Catalyst','Imine','Thiol']):
            mol = data[category][i]
            mol_img = Image.open('./' + category + '\\' + dicts[index][mol] + '.png')
            mol_data = np.array(transform(mol_img))
            test_reaction = np.concatenate([test_reaction,mol_data],1)
        test_label.append(data['Output'][i])
        test_fea.append(test_reaction)
    
    feaTest = torch.from_numpy(np.array(test_fea))
    targetsTest = torch.from_numpy(np.array(test_label).flatten())
    data_loader = torch.utils.data.TensorDataset(feaTest,targetsTest)
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
            nn.Linear(435080, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
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
batch_size = 4
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
])

data = pd.read_csv(file)

Catalyst = get_molecule_dict('Catalyst')
Imine = get_molecule_dict('Imine')
Thiol = get_molecule_dict('Thiol')

# Catalyst_loader = generate_dataloader_single('Catalyst',Catalyst)
# Imine_loader = generate_dataloader_single('Imine',Imine)
# Thiol_loader = generate_dataloader_single('Thiol',Thiol)

# train_loader = generate_dataloader_total()
train_loader = generate_traindataloader_cat()
test_loader = generate_testdataloader_cat()

model = CNN()
model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

mae_train_total = []
mae_test_total = []
for epoch in range(epochs):
    print(epoch+1)
    mae_train = []
    model.train()
    for batch_idx,(fea,label) in enumerate(train_loader):
        fea,label = fea.to(DEVICE),label.to(DEVICE)
        optimizer.zero_grad()
        output = model(fea.to(torch.float))
        output = output.squeeze(-1)
        loss = squared_loss(output, label)
        loss.backward()
        optimizer.step()
        pred = output
        mae_train.append(mean_absolute_error((label.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy()))
        # mae_train.append(mean_absolute_error(label.detach().numpy(), pred.detach().numpy()))
        
    mae_train = np.array(mae_train).mean()
    mae_train_total.append(mae_train)
    if (epoch+1) % 10 == 0:
        print('------------------------------------------')
        print('开始训练第{}轮'.format(epoch+1))
        print('mae_train:{}'.format(mae_train))

    # if (epoch+1) == EPOCHS:
    #     target = np.array((target.to(torch.float)).data.cpu().numpy()).reshape(-1,1)
    #     pred = np.array(pred.data.cpu().numpy()).reshape(-1,1)
    #     # target = np.array(target).reshape(-1,1)
    #     # pred = np.array(pred).reshape(-1,1)
    #     target_pred_train = np.concatenate((target, pred),axis = 1)
    #     np.savetxt('FMSD_G_subset_train_target_pred_'+ 'test_sub'+ '.csv',target_pred_train,delimiter=',')
    #     np.savetxt('FMSD_G_subset_r2_train_'+ 'test_sub' + '.csv',mae_train_total,delimiter=',')

    # Validation of the model.
    model.eval()
    
    mae_test = []
    with torch.no_grad():
        for batch_idx, (fea, label) in enumerate(test_loader):
            # if batch_idx * len(fea) >= len(fea):
            #     break
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(fea.to(torch.float))
            pred = output
            mae_test.append(mean_absolute_error((target.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy()))
            # mae_test.append(mean_absolute_error(label.detach().numpy(), pred.detach().numpy()))

    mae_test = np.array(mae_test).mean()
    mae_test_total.append(mae_test)
    if (epoch+1) % 10 == 0:
        print('mae_test:{}'.format(mae_test))
        print('------------------------------------------')

