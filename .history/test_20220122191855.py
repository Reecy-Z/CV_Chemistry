import os
import torch
print(torch.cuda.is_available())#是否有可用的gpu
print(torch.cuda.device_count())#有几个可用的gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#声明gpu
dev=torch.device('cuda:0')#调用哪个gpu
a=torch.rand(100,100).to(dev)
