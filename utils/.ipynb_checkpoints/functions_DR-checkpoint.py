import torch
from torch import nn
from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

class MyDataset(Dataset):
    def __init__(self, img, label, width):
        self.data = img
        self.label = torch.Tensor(label)
        self.value = torch.Tensor(width)
        
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index].long()
        y2 = self.value[index].float()
        return x, y, y2

    def __len__(self):
        return len(self.data) 
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape) # row, col
    
    def __getsize__(self):
        return (self.__len__())
    
def caps_loss(y_true, y_pred, x, x_recon, lam_recon, y, weight):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    y_true = y_true.squeeze(1)
    y_pred = y_pred.squeeze(1)
    x = x.squeeze(1)
    x_recon = x_recon.squeeze(1)
    
    L_margin = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2 
    L_recon = nn.MSELoss()(x_recon.view(-1, 3, 256, 256), x)
   
    return L_margin.mean() + lam_recon * L_recon

def ex_caps_loss(y_true, y_pred, x, x_recon, lam_recon, y, weight):
    """
    Weighted Margin Loss
    """
    y_true = y_true.squeeze(1)
    y_pred = y_pred.squeeze(1)
    x = x.squeeze(1)
    x_recon = x_recon.squeeze(1)
    
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2 
    L_margin = (L.sum(dim=1)*(((y == 1) * weight[1]) + ((y==0)*weight[0]))).mean()
    L_recon = nn.MSELoss()(x_recon.view(-1, 3, 256, 256), x)
   
    return L_margin + lam_recon * L_recon


def train_DR(args, train_loader, valid_loader, model, criterion, optimizer, weight1, result_folder):
    train_loss_list = []
    valid_loss_list = []
    valid_acc_list=[]
    
    if args.model == 'originalDR' or args.model=='Efficient':
        print("original caps loss")
        criterion = caps_loss
    else:
        print("weighted caps loss")
        criterion = ex_caps_loss
    
    def test_DR(args, valid_loader, model, criterion, optimizer):
        model = model.eval()
        reconstruction = [] 
        predictions = [] 
        prob= []
        answers = [] # recon target
        labels = [] # pred target 
        valid_loss = []
        with torch.no_grad():
            for (x, y, y2) in tqdm(valid_loader, leave=False):
                x = x.float().cuda()
                y = y.float().cuda()
                y2 = y2.cuda()

                answers.extend(x.squeeze().detach().cpu().numpy())
                labels.extend(y.detach().cpu().numpy())

                y_onehot = torch.zeros(y.size(0), 2).scatter_(1, torch.Tensor(y.view(-1,1).detach().cpu()).type(torch.int64), 1.)  
                y_onehot = y_onehot.cuda()
                
                if args.model == 'Efficient':
                    classes, recon = model(x)
                else:
                    classes, recon, width = model(x)

                loss = criterion(y_onehot, classes, x[:, :3, : ,:], recon, 0.0005*3*256*256, y, weight1)
                reconstruction.extend(recon.squeeze().detach().cpu().numpy())
                predictions.extend(torch.max(classes,1)[1].detach().cpu().numpy())
                prob.extend(classes.detach().cpu().numpy())
                valid_loss.append(loss.item())   
                
        return valid_loss, labels, predictions, prob

    best_loss = 10 ** 9 
    patience_limit = 5 
    patience_check = 0 

    for epoch in range(0, args.num_epoch):
        model = model.train()

        train_loss = []
        for (x, y, y2) in tqdm(train_loader, leave=False):
            x = x.float().cuda()
            y = y.float().cuda()
            y2 = y2.cuda()

            y_onehot = torch.zeros(y.size(0), 2).scatter_(1, torch.Tensor(y.view(-1,1).detach().cpu()).type(torch.int64), 1.) 
            y_onehot = y_onehot.cuda()

            optimizer.zero_grad()
            if args.model == 'Efficient':
                classes, recon = model(x)
            else:
                classes, recon, width = model(x)

            loss = criterion(y_onehot, classes, x[:, :3, : ,:], recon, 0.0005*3*256*256, y, weight1)
            if args.model == 'CardioCaps':
                loss = loss + nn.MSELoss()(width.squeeze(), y2) * args.decay

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        
        valid_loss, labels, predictions, prob = test_DR(args, valid_loader, model, criterion, optimizer)

        valid_loss = np.mean(valid_loss)
        valid_acc = accuracy_score(labels, predictions)
        valid_acc_list.append(valid_acc)

        print("epoch: {}/{} | trn_loss: {:.4f} | val_loss: {:.4f} / val_acc:{:.4f} ".format(
                    epoch, args.num_epoch, train_loss, valid_loss, valid_acc
                ))
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        if (epoch==0) or (epoch>0 and (min(valid_loss_list[:-1])>valid_loss_list[-1])):
            torch.save({
                'epoch': epoch,
                'loss' : valid_loss_list[-1],
                'acc' : valid_acc_list[-1],
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'criterion' : caps_loss
            }, os.path.join(result_folder, f'dr-capsule-ecg-best.pt'))

        if valid_loss > best_loss:
            patience_check += 1
            if patience_check >= patience_limit: 
                break
        else:
            best_loss = valid_loss
            patience_check = 0
        
    print("Finished training...")
    print("Drawing loss plot...")
    fig, ax = plt.subplots(1,2,figsize=(20,5))

    ax0 = ax[0]
    ax0.plot(train_loss_list, c= 'blue')
    ax0.plot(valid_loss_list, c='red')

    ax1 = ax[1]
    ax1.plot(valid_loss_list, c='red', marker='o')

    fig.suptitle("Loss", fontsize=15)
    plt.savefig(os.path.join(result_folder, 'dr_loss.png'))
    pd.DataFrame([train_loss_list,valid_loss_list]).to_csv(os.path.join(result_folder, f'dr_loss.csv'), index=0)
    
    
def plot_confusion_matrix(y_label, y_pred, result_folder, normalized=True):
    mat = confusion_matrix(y_label, y_pred)
    cmn = mat.astype('float')
    if normalized:
        cmn = cmn / mat.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues', cbar=False, annot_kws={"fontsize":25}) 
    plt.ylabel('Actual', fontsize=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.title("Confusion Matrix (%)", fontsize=15)
    plt.savefig(os.path.join(result_folder, 'dr_confmatrix.pdf'))
    
    
def inference_DR(args, test_loader, model, criterion, optimizer, weight1, result_folder):
    print("Inference...")
    if args.model == 'originalDR' or args.model=='Efficient':
        print("original caps loss")
        criterion = caps_loss
    else:
        print("weighted caps loss")
        criterion = ex_caps_loss
    
    checkpoint = torch.load(os.path.join(result_folder, 'dr-capsule-ecg-best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(checkpoint['epoch'], checkpoint['loss'], checkpoint['acc'])
    
    reconstruction = []
    predictions = []
    prob= []
    answers = []
    labels = []
    test_loss = []
    mse_loss = []
    with torch.no_grad():
        for (x, y, y2) in tqdm(test_loader, leave=False):
            x = x.float().cuda()
            y = y.float().cuda()
            y2 = y2.cuda()

            answers.extend(x.squeeze().detach().cpu().numpy())
            labels.extend(y.detach().cpu().numpy())

            y_onehot = torch.zeros(y.size(0), 2).scatter_(1, torch.Tensor(y.view(-1,1).detach().cpu()).type(torch.int64), 1.)  
            y_onehot = y_onehot.cuda()
            if args.model == 'Efficient':
                classes, recon = model(x)
            else:
                classes, recon, width = model(x)

            loss = criterion(y_onehot, classes, x[:, :3, : ,:], recon, 0.0005*3*256*256, y, weight1)
            if args.model == 'CardioCaps':
                loss = loss + nn.MSELoss()(width.squeeze(), y2) * args.decay
        
            reconstruction.extend(recon.squeeze().detach().cpu().numpy())
            predictions.extend(torch.max(classes,1)[1].detach().cpu().numpy())
            prob.extend(classes.detach().cpu().numpy())
            test_loss.append(loss.item()) 

    test_loss = np.mean(test_loss)
    test_acc = accuracy_score(labels, predictions)
    f1score = f1_score(labels, predictions, average='weighted')
    roc = roc_auc_score(labels, predictions)
    pr = precision_score(labels, predictions)
    
    plot_confusion_matrix(labels, predictions, result_folder)
    
    print (f'Performance: {test_acc} / {f1score} / {roc} / {pr}')
    
    result_label = pd.DataFrame([labels, predictions])
    result_pred = pd.DataFrame([predictions, prob])

    result_label.to_csv(os.path.join(result_folder, f'classi_pred_label.csv'), index=0)
    result_pred.to_csv(os.path.join(result_folder, f'classi_pred.csv'), index=0)
    
    return test_acc, f1score, roc, pr