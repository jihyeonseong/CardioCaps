import os
import random
import numpy as np
import pandas as pd
import argparse

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import torch
from torch import nn
import torchvision

from utils.functions_CNN import MyDataset, train_CNN, inference_CNN
from model.CNN import CNN
from model.CNN2 import CNN2
from model.Unet import UNet
from model.ViT import SimpleViT
from model.EMCaps import CapsNet, SpreadLoss

def main(args):
    print("Read Dataset...")
    with open(os.path.join(args.data_folder, 'train_x.npy'), 'rb') as f:
        train_x = np.load(f)
    with open(os.path.join(args.data_folder, 'train_y.npy'), 'rb') as f:
        train_y = np.load(f)
    
    print("Split train/validation/test...")
    X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, stratify=train_y, test_size=0.4)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, stratify=y_valid, test_size=0.2)
    
    print("Make Dataset...")
    train_dataset = MyDataset(X_train, y_train)
    valid_dataset = MyDataset(X_valid, y_valid)
    test_dataset = MyDataset(X_test, y_test)
    
    print("Make DataLoader...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, num_workers=4, pin_memory=True, shuffle=True) 
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False, num_workers=4, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, num_workers=4, pin_memory=True, shuffle=False)
    
    result_folder = os.path.join(args.result_folder, f'{args.model}_{args.seed}/classification')
    os.makedirs(result_folder, exist_ok = True)
    
    if args.model == 'CNN':
        print("Model: CNN")
        model = CNN(in_features=9,
                    out_features=2,
                    pool_size=3, 
                    hidden_dim=32, 
                    capsule_num=2).cuda()
        
    elif args.model == 'CNN2':
        print("Model: CNN-v2")
        model = CNN2(in_features=9,
                    out_features=2,
                    pool_size=3, 
                    hidden_dim=32, 
                    capsule_num=2).cuda()
        
    elif args.model == 'ResNet':
        print("Model: ResNet18")
        model = torchvision.models.resnet18()
        model.conv1 = nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.cuda()
        
    elif args.model == 'UNet':
        print("Model: Unet")
        model = UNet(in_features=9, 
                     out_features=2,
                     hidden_dim=32).cuda()
        
    elif args.model == 'ViT':
        print("Model: ViT")
        model = SimpleViT(image_size=(256,256), patch_size=4, num_classes=2, dim=16, depth=2, heads=4, mlp_dim=16).cuda()
        
    elif args.model == 'EM':
        print("Model: EMCaps")
        model = CapsNet(in_features=9,
                        out_features=2).cuda()
        
    else:
        raise Exception("No Model!")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    class_weight_vec1 = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train), y=y_train)
    weight1 = torch.Tensor(np.unique(class_weight_vec1)).cuda()
    criterion = nn.CrossEntropyLoss(weight=weight1)
    
    if args.model == 'EM':
        criterion = SpreadLoss(num_class=2, m_min=0.2, m_max=0.9)
        
    if args.train == True:
        train_CNN(args, train_loader, valid_loader, model, criterion, optimizer, result_folder)
    
    if args.inference == True:
        acc, f1, roc, pr = inference_CNN(args, test_loader, model, criterion, optimizer, result_folder)
        pd.DataFrame([acc, f1, roc, pr], index=['acc', 'f1', 'roc', 'pr']).to_csv(os.path.join(result_folder, 'perf.csv'))
    
    print("Finished!")
    
    
if __name__=='__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpuidx', default=2, type=int, help='gpu index')
    parser.add_argument('--seed', default=10, type=int, help='seed test')
    
    parser.add_argument('--data_folder', default='./data', type=str, help='data path')
    parser.add_argument('--result_folder', default='./check', type=str, help='result path')
    
    parser.add_argument('--model', default='CNN', type=str, help='model type')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_epoch', default=100, type=int, help='learning epochs')
    
    parser.add_argument('--train', action='store_true', help='training')
    parser.add_argument('--inference', action='store_true', help='test')
    
    args = parser.parse_args()
    
    random_seed=args.seed
    torch.manual_seed(random_seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed) 
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuidx)
    
    main(args)