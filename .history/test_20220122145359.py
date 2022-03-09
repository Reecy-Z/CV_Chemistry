import shutil
import os,sys
import cv2
from PIL import Image
import numpy as np

input_path = '.\Catalyst'
out_path ='/data/craft/tc/train/dan'


filelist = os.listdir(input_path)
for item in filelist:        
    img_path = os.path.join(input_path, item)
    move_path =os.path.join(out_path, item)
    
    
    image = Image.open(img_path)
    image = np.array(image)
    print(item)
    #print('image',image)
    #print('image.size',image.size)
    print('image.shape',image.shape)
    
    
    # #2就是单通道图片
    # if len(image.shape) == 2:    
    #     shutil.move(img_path, move_path)
    # else:
    #     continue
    

