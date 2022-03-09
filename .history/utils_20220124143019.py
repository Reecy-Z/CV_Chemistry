import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def squared_loss(y_true, y_pred):
    """Compute the squared loss for regression.
    """
    return ((y_true - y_pred) ** 2).mean() / 2

def get_molecule_dict(file,category):
    data = pd.read_csv(file)
    category_dict = {}
    i = 1
    for mol in data[category]:
        if mol not in category_dict.keys():
            category_dict[mol] = category + '_' +str(i)
            i += 1
    return category_dict

def transform_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
    ])
    img_data = transform(image)
    return img_data

def generate_traindataloader_cat(file,split,dicts,batch_size,dir):
    data = pd.read_csv(file)
    columns = data.columns.values.tolist()[:-1]
    train_fea = []
    train_label = []
    for i in range(split):
        train_reaction = np.empty((3,300,300))
        print(i)
        for index,category in enumerate(columns):
            mol = data[category][i]
            mol_img = Image.open('./' + dir + '\\' + category + '\\' + dicts[index][mol] + '.png')
            mol_data = np.array(transform_image(mol_img))
            train_reaction = np.concatenate([train_reaction,mol_data],1)
        train_label.append(data['Output'][i])
        train_fea.append(train_reaction)
    
    feaTrain = torch.from_numpy(np.array(train_fea))
    targetsTrain = torch.from_numpy(np.array(train_label).flatten())
    data_loader = torch.utils.data.TensorDataset(feaTrain,targetsTrain)
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size = batch_size, shuffle=True)

    return data_loader

def generate_testdataloader_cat(file,split,end,dicts,batch_size):
    data = pd.read_csv(file)
    columns = data.columns.values.tolist()[:-1]
    test_fea = []
    test_label = []
    for i in range(split,end):
        test_reaction = np.empty((3,300,300))
        print(i)
        for index,category in enumerate(columns):
            mol = data[category][i]
            mol_img = Image.open('./' + dir + '\\' + category + '\\' + dicts[index][mol] + '.png')
            mol_data = np.array(transform_image(mol_img))
            test_reaction = np.concatenate([test_reaction,mol_data],1)
        test_label.append(data['Output'][i])
        test_fea.append(test_reaction)
    
    feaTest = torch.from_numpy(np.array(test_fea))
    targetsTest = torch.from_numpy(np.array(test_label).flatten())
    data_loader = torch.utils.data.TensorDataset(feaTest,targetsTest)
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size = batch_size, shuffle=True)

    return data_loader




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