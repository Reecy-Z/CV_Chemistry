import numpy as np
import pandas as pd
import torch
import utils
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from PIL import Image
from torch import nn
from sklearn.metrics import mean_absolute_error

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
 
        self.fc = nn.Sequential(
            nn.Linear(65262, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
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
end = 1075
learning_rate = 0.002
epochs = 1000
batch_size = 20
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
])

data = pd.read_csv(file)

Catalyst = utils.get_molecule_dict('Catalyst',file)
Imine = utils.get_molecule_dict('Imine',file)
Thiol = utils.get_molecule_dict('Thiol',file)

train_loader = generate_traindataloader_cat()
test_loader = generate_testdataloader_cat()

model = CNN()
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        loss = utils.squared_loss(output, label)
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
            fea, label = fea.to(DEVICE), label.to(DEVICE)
            output = model(fea.to(torch.float))
            pred = output
            mae_test.append(mean_absolute_error((label.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy()))
            # mae_test.append(mean_absolute_error(label.detach().numpy(), pred.detach().numpy()))

    mae_test = np.array(mae_test).mean()
    mae_test_total.append(mae_test)
    if (epoch+1) % 10 == 0:
        print('mae_test:{}'.format(mae_test))
        print('------------------------------------------')

