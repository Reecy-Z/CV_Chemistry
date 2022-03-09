import numpy as np
import torch
import utils
import torch.optim as optim
import models
from sklearn.metrics import mean_absolute_error,r2_score

Dir = './18 science'
file = 'test_3.csv'
split = 3059
end = 3955
learning_rate = 0.002
epochs = 1000
batch_size = 20
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 19 science
# Catalyst = utils.get_molecule_dict('Catalyst',file)
# Imine = utils.get_molecule_dict('Imine',file)
# Thiol = utils.get_molecule_dict('Thiol',file)

# 18 science
Ligand = utils.get_molecule_dict(Dir,file,'Ligand')
Additive = utils.get_molecule_dict(Dir,file,'Additive')
Base = utils.get_molecule_dict(Dir,file,'Base')
Aryl_halide = utils.get_molecule_dict(Dir,file,'Aryl halide')
dicts = [Ligand,Additive,Base,Aryl_halide]

train_fea, train_label = utils.generate_Traindata_loader(file,split,dicts,batch_size,Dir)
test_fea, test_label = utils.generate_Testdata_loader(file,split,end,dicts,batch_size,Dir)

model = models.CNN()
# model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

mae_train_total = []
r2_train_total = []
mae_test_total = []
r2_test_total = []
for epoch in range(epochs):
    print(epoch+1)
    mae_train = []
    r2_train = []
    model.train()
    for batch_idx,fea in enumerate(train_fea):
        label = train_label[batch_idx]
        fea,label = torch.from_numpy(fea),torch.from_numpy(label)
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
        r2_train.append(r2_score((label.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy()))
        
    mae_train = np.array(mae_train).mean()
    mae_train_total.append(mae_train)
    r2_train = np.array(r2_train).mean()
    r2_train_total.append(r2_train)

    if (epoch+1) % 10 == 0:
        print('------------------------------------------')
        print('开始训练第{}轮'.format(epoch+1))
        print('mae_train:{}'.format(mae_train))
        print('r2_train:{}'.format(r2_train))


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
    r2_test = []
    with torch.no_grad():
        for batch_idx, (fea, label) in enumerate(test_loader):
            # if batch_idx * len(fea) >= len(fea):
            #     break
            fea, label = fea.to(DEVICE), label.to(DEVICE)
            output = model(fea.to(torch.float))
            pred = output
            mae_test.append(mean_absolute_error((label.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy()))
            # mae_test.append(mean_absolute_error(label.detach().numpy(), pred.detach().numpy()))
            r2_test.append(r2_score((label.to(torch.float)).data.cpu().numpy(), pred.data.cpu().numpy()))

    mae_test = np.array(mae_test).mean()
    mae_test_total.append(mae_test)
    r2_test = np.array(r2_test).mean()
    r2_test_total.append(r2_test)

    if (epoch+1) % 10 == 0:
        print('mae_test:{}'.format(mae_test))
        print('r2_test:{}'.format(r2_test))
        print('------------------------------------------')

