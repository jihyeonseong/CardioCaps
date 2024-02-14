import cv2
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import torch
from torchvision import transforms
from PIL import Image

def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True

if __name__=='__main__':
    file_list = []
    file_name = []
    for i in range(1,4):
        path = f"/data/Batch{i}/*"
        file_list.append(glob.glob(path))
        path = f"/data/Batch{i}/"
        file_name.append(os.listdir(path))
        
    os.makedirs('/data/img', exist_ok=True)
    
    for i, (file, name) in enumerate(zip(file_list, file_name)):
        print("Batch: ", i)
        for file_path, file_name_ in zip(file, name):
            #convert_avi_to_mp4(file_path, f'/data3/jihyeon/CapsUltra/data/mp4/{file_name}')
            vidcap = cv2.VideoCapture(f'{file_path}')
            success,image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite(f"/data3/jihyeon/CapsUltra/data/img/{file_name_}_%06d.jpg" % count, image)     # save frame as JPEG file
                success,image = vidcap.read()
                count += 1
                
    path = f"/data/img/*"
    path_list = glob.glob(path)
    path = f"/data/img/"
    file_name = os.listdir(path)
    
    label = pd.read_csv('/data/MeasurementsList.csv', index_col=0)
    le = LabelEncoder()
    label['Calc_n'] = le.fit_transform(label['Calc'])
    
    convert_tensor = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(256,256))])
    convert_tensor2 = transforms.Compose([transforms.ToTensor(), transforms.Resize((728,728)), 
                                         transforms.RandomRotation(10), transforms.RandomCrop(512), transforms.CenterCrop(256)])
    convert_tensor3 = transforms.Compose([transforms.ToTensor(), transforms.Resize((728,728)), 
                                         transforms.RandomRotation(10), transforms.RandomCrop(512), transforms.CenterCrop(256)])

    train_x = []
    train_y = []
    width_y = []

    class_num_l = [0,0,0,0,0,0]
    for i in tqdm(range(11464, 11464+300)):
        name = label.iloc[i, 0]
        class_ = label.iloc[i, 1] 
        value_ = label.iloc[i, 2]
        class_num = label.iloc[i, -1]

        if class_!=1 and class_!=5:
            data_list = []
            for f in file_name:
                if name in f:
                    data_list.append(f)
                    #break
            data_list = sorted(data_list)[:50]

            if len(data_list) != 0 and class_ in ['IVSd', 'IVSs', 'LVPWd', 'LVPWs', 'LVIDd', 'LVIDs', 'LVID', 'RVDd']:
                for k, data in enumerate(data_list):
                    mask  = torch.rand((256, 256)) < 0.5
                    img0 = convert_tensor(Image.open(os.path.join('/data/img',data_list[k])))
                    img = convert_tensor2(Image.open(os.path.join('/data/img',data_list[k])))

                    img2 = convert_tensor3(Image.open(os.path.join('/data/img',data_list[k])))
                    img = torch.cat((img0, img, img2), dim=0)

                    train_x.append(img.numpy())
                    width_y.append(value_)
                    if class_ == 'IVSd' or class_ == 'IVSs' or class_ == 'LVPWd' or class_ == 'LVPWs':
                        if value_ < 1.1:
                            train_y.append(0)
                        else:
                            train_y.append(1)
                    elif class_ == 'LVIDd' or class_ == 'LVIDs':
                        if value_ < 5.6:
                            train_y.append(0)
                        else:
                            train_y.append(1)
                    elif class_ == 'LVID':
                        if value_ <4:
                            train_y.append(0)
                        else:
                            train_y.append(1)
                    elif class_ == 'RVDd':
                        if value < 2.3:
                            train_y.append(0)
                        else:
                            train_y.append(1)

            if i % 100 == 0:
                with open('/data/train_x.npy', 'wb') as f:
                    np.save(f, np.array(train_x))
                with open('/data/train_y.npy', 'wb') as f:
                    np.save(f, np.array(train_y))
                with open('/data/width_y.npy', 'wb') as f:
                    np.save(f, np.array(width_y))

        class_num_l[class_num] +=1
        
    with open('/data/train_x.npy', 'wb') as f:
        np.save(f, np.array(train_x, dtype=np.float16))
    with open('/data/train_y.npy', 'wb') as f:
        np.save(f, np.array(train_y, dtype=np.float16))
    with open('/data/width_y.npy', 'wb') as f:
        np.save(f, np.array(width_y, dtype=np.float16))