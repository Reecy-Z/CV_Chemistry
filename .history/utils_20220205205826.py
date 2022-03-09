from cgi import test
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from xgboost import train

def squared_loss(y_true, y_pred):
    """Compute the squared loss for regression.
    """
    return ((y_true - y_pred) ** 2).mean() / 2

def get_molecule_dict(Dir,file,category):
    data = pd.read_csv('./' + Dir + '\\' + file)
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

def generate_traindataloader_cat(file,split,dicts,batch_size,Dir):
    data = pd.read_csv('./' + Dir + '\\' + file)
    columns = data.columns.values.tolist()[:-1]
    train_fea = []
    train_label = []
    for i in range(split):
        print(i)
        for index,category in enumerate(columns):
            mol = data[category][i]
            mol_img = Image.open('./' + Dir + '\\' + category + '\\' + dicts[index][mol] + '.png')
            mol_data = np.array(transform_image(mol_img))
            if index == 0:
                train_reaction = mol_data
            else:
                train_reaction = np.concatenate([train_reaction,mol_data],1)
        train_label.append(data['Output'][i])
        train_fea.append(train_reaction)
    
    feaTrain = torch.from_numpy(np.array(train_fea))
    targetsTrain = torch.from_numpy(np.array(train_label).flatten())
    data_loader = torch.utils.data.TensorDataset(feaTrain,targetsTrain)
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size = batch_size, shuffle=True)

    return data_loader

def generate_testdataloader_cat(file,split,end,dicts,batch_size,Dir):
    data = pd.read_csv('./' + Dir + '\\' + file)
    columns = data.columns.values.tolist()[:-1]
    test_fea = []
    test_label = []
    for i in range(split,end):
        print(i)
        for index,category in enumerate(columns):
            mol = data[category][i]
            mol_img = Image.open('./' + Dir + '\\' + category + '\\' + dicts[index][mol] + '.png')
            mol_data = np.array(transform_image(mol_img))
            if index == 0:
                test_reaction = mol_data
            else:
                test_reaction = np.concatenate([test_reaction,mol_data],1)
        test_label.append(data['Output'][i])
        test_fea.append(test_reaction)
    
    feaTest = torch.from_numpy(np.array(test_fea))
    targetsTest = torch.from_numpy(np.array(test_label).flatten())
    data_loader = torch.utils.data.TensorDataset(feaTest,targetsTest)
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size = batch_size, shuffle=True)

    return data_loader

def generate_Traindata_loader(file,split,dicts,batch_size,Dir):
    data = pd.read_csv('./' + Dir + '\\' + file)
    columns = data.columns.values.tolist()[:-1]
    train_data = []
    train_data_batch = ([],[])

    # 存在最后几个不能达到一个batch的情况
    num = split//batch_size

    for i in range(split):
        print(i)
        train_reaction_fea = []
        train_reaction_label = []
        for index,category in enumerate(columns):
            mol = data[category][i]
            mol_img = Image.open('./' + Dir + '\\' + category + '\\' + dicts[index][mol] + '.png')
            mol_data = np.array(transform_image(mol_img))
            train_reaction_fea.append(mol_data)
        train_reaction_label.append(data['Output'][i])
        train_data_batch[0].append(train_reaction_fea)
        train_data_batch[1].append(train_reaction_label)
        train_fea_batch.append(train_reaction_fea)
        train_label_batch.append(train_reaction_label)

        train_reaction = (train_reaction_fea,train_reaction_label)
        train_data_batch.append(train_reaction)
        if len(train_data_batch) == batch_size:
            train_data.append(np.array(train_data_batch))
            train_data_batch = []
        elif (len(train_data) == num) and ((split%batch_size) != 0):
            if len(train_data_batch) == split - len(train_data)*batch_size:
                train_data.append(np.array(train_data_batch))
                break
            else:
                continue

    return train_data

def generate_Testdata_loader(file,split,end,dicts,batch_size,Dir):
    data = pd.read_csv('./' + Dir + '\\' + file)
    columns = data.columns.values.tolist()[:-1]
    test_data = []
    test_data_batch = []

    # 存在最后几个不能达到一个batch的情况
    num = (end-split)//batch_size

    for i in range(split,end):
        print(i)
        test_reaction_fea = []
        for index,category in enumerate(columns):
            mol = data[category][i]
            mol_img = Image.open('./' + Dir + '\\' + category + '\\' + dicts[index][mol] + '.png')
            mol_data = np.array(transform_image(mol_img))
            test_reaction_fea.append(mol_data)
        test_reaction = (test_reaction_fea,data['Output'][i])
        test_data_batch.append(test_reaction)
        if len(test_data_batch) == batch_size:
            test_data.append(np.array(test_data_batch))
            test_data_batch = []
        elif (len(test_data) == num) and (((end-split)%batch_size) != 0):
            if len(test_data_batch) == (end-split) - len(test_data)*batch_size:
                test_data.append(np.array(test_data_batch))
                break
            else:
                continue

    return test_data

#训练时存在问题
def generate_dataloader_single(file,category,category_dict,split,batch_size):
    data = pd.read_csv(file)
    train_fea = []
    train_label = []
    for i in range(split):
        print(i)
        mol = data[category][i]
        mol_img = Image.open('./' + category + '\\' + category_dict[mol] + '.png')
        mol_data = transform_image(mol_img)
        train_fea.append(np.array(mol_data))
        train_label.append(data['Output'][i])
    
    feaTrain = torch.from_numpy(np.array(train_fea))
    targetsTrain = torch.from_numpy(np.array(train_label))
    data_loader = torch.utils.data.TensorDataset(feaTrain,targetsTrain)
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size = batch_size, shuffle=True)

    return data_loader

def generate_dataloader_total(file,dicts,split,batch_size):
    data = pd.read_csv(file)
    train_fea = []
    train_label = []
    for i in range(split):
        train_reaction = []
        print(i)
        for index,category in enumerate(['Catalyst','Imine','Thiol']):
            mol = data[category][i]
            mol_img = Image.open('./' + category + '\\' + dicts[index][mol] + '.png')
            mol_data = transform_image(mol_img)
            train_reaction.append(np.array(mol_data))
        train_label.append(data['Output'][i])
        train_fea.append(train_reaction)
    
    feaTrain = torch.from_numpy(np.array(train_fea))
    targetsTrain = torch.from_numpy(np.array(train_label))
    data_loader = torch.utils.data.TensorDataset(feaTrain,targetsTrain)
    data_loader = torch.utils.data.DataLoader(data_loader, batch_size = batch_size, shuffle=True)

    return data_loader