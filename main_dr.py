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

from utils.functions_DR import MyDataset, train_DR, inference_DR
from model.DR import CapsuleNet
from model.EffCaps import EfficientCapsNet, ReconstructionNet, EfficientCapsNetWithReconstruction

def main(args):
    print("Read Dataset...")
    with open(os.path.join(args.data_folder, 'train_x.npy'), 'rb') as f:
        train_x = np.load(f)
    with open(os.path.join(args.data_folder, 'train_y.npy'), 'rb') as f:
        train_y = np.load(f)
    with open(os.path.join(args.data_folder, 'width_y.npy'), 'rb') as f:
        width_y = np.load(f)
    
    print("Split train/validation/test...")
    X_train, X_valid, y_train, y_valid, width_yt, width_yv = train_test_split(train_x, train_y, width_y, stratify=train_y, test_size=0.4)
    X_valid, X_test, y_valid, y_test, width_yv, width_ytt = train_test_split(X_valid, y_valid, width_yv, stratify=y_valid, test_size=0.2)
    
    print("Make Dataset...")
    train_dataset = MyDataset(X_train, y_train, width_yt)
    valid_dataset = MyDataset(X_valid, y_valid, width_yv)
    test_dataset = MyDataset(X_test, y_test, width_ytt)
    
    print("Make DataLoader...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, num_workers=4, pin_memory=True, shuffle=True) 
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, drop_last=False, num_workers=4, pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, num_workers=4, pin_memory=True, shuffle=False)
    
    model = CapsuleNet(args, [9,256,256], 2, 3, None, None).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    class_weight_vec1 = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train), y=y_train)
    weight1 = torch.Tensor(np.unique(class_weight_vec1)).cuda()
    criterion = nn.MSELoss()

    if args.model == 'originalDR':
        print("Model: originalDR")
        result_folder = os.path.join(args.result_folder, f'originalDR{int(args.dr)}_{args.affine}_{args.seed}/classification')
        os.makedirs(result_folder, exist_ok = True)
        
    elif args.model == 'Efficient':
        print("Model: Efficient CapsNet")
        model = EfficientCapsNet(in_features=9,
                                 out_features=2).cuda()
        reconstruction_model = ReconstructionNet(16, 2)
        reconstruction_alpha = 0.0005
        model = EfficientCapsNetWithReconstruction(model, reconstruction_model).cuda()
        result_folder = os.path.join(args.result_folder, f'EfficientCaps_{args.seed}/classification')
        os.makedirs(result_folder, exist_ok = True)
        
    elif args.model == 'CardioCaps':
        print("Model: CardioCaps")
        result_folder = os.path.join(args.result_folder, f'ourDR{int(args.dr)}_{args.affine}_decay{args.decay}_{args.seed}/classification')
        os.makedirs(result_folder, exist_ok = True)
        
    else:
        raise Exception("No Model!")
    
    if args.train == True:
        train_DR(args, train_loader, valid_loader, model, criterion, optimizer, weight1, result_folder)
    
    if args.inference == True:
        acc, f1, roc, pr = inference_DR(args, test_loader, model, criterion, optimizer, weight1, result_folder)
        pd.DataFrame([acc, f1, roc, pr], index=['acc', 'f1', 'roc', 'pr']).to_csv(os.path.join(result_folder, 'perf.csv'))
        
    print("Finished!")
        
        
if __name__=='__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpuidx', default=2, type=int, help='gpu index')
    parser.add_argument('--seed', default=10, type=int, help='seed test')
    
    parser.add_argument('--data_folder', default='./data', type=str, help='data path')
    parser.add_argument('--result_folder', default='./check', type=str, help='result path')
    
    parser.add_argument('--model', default='originalDR', type=str, help='model type')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--num_epoch', default=100, type=int, help='learning epochs')
    parser.add_argument('--decay', default=0.05, type=float, help='regression decay')
    
    parser.add_argument('--affine', default='param', type=str, help='affine matrix type')
    parser.add_argument('--dr', action='store_true', help='dynamic routing / self attention')
    
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