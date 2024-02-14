import torch
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
    def __init__(self, img, label):
        self.data = img
        self.label = torch.Tensor(label)
        
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index].long()
        return x, y

    def __len__(self):
        return len(self.data) 
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape) # row, col
    
    def __getsize__(self):
        return (self.__len__())
    
    
def train_CNN(args, train_loader, valid_loader, model, criterion, optimizer, result_folder):
    def test_CNN(args, valid_loader, model, criterion, optimizer):
        model = model.eval()
        predictions = []
        answers = []
        valid_loss = []
        prob = []
        with torch.no_grad():
            for (x, y) in tqdm(valid_loader, leave=False):
                x = x.float().cuda()
                y = y.cuda()
                answers.extend(y.detach().cpu().numpy())

                outputs = model(x).squeeze(1)
                if args.model == 'EM':
                    loss = criterion(outputs, y, r=1)
                else:
                    loss = criterion(outputs, y) 

                prob.extend(torch.nn.Softmax()(outputs).detach().cpu().numpy())
                predictions.extend(torch.max(torch.nn.Softmax()(outputs),1)[1].detach().cpu().numpy())
                valid_loss.append(loss.item())
        
        return valid_loss, answers, predictions, prob

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list=[]
    
    best_loss = 10 ** 9 
    patience_limit = 5 
    patience_check = 0 
    
    for epoch in range(0, args.num_epoch):
        model = model.train()

        train_loss = []
        predictions = []
        answers = []
        prob= []
        batch_idx=0
        train_len = len(train_loader)
        for (x, y) in tqdm(train_loader, leave=False):
            x = x.float().cuda()
            y = y.cuda()
            answers.extend(y.detach().cpu().numpy())

            optimizer.zero_grad()
            outputs = model(x).squeeze(1)
            #print(outputs.shape)
            
            if args.model == 'EM':
                r = (1.*batch_idx + (epoch-1)*train_len) / (args.num_epoch*train_len)
                loss = criterion(outputs, y, r)
                batch_idx += 1
            else:
                loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            prob.extend(torch.nn.Softmax()(outputs).detach().cpu().numpy())
            predictions.extend(torch.max(torch.nn.Softmax()(outputs),1)[1].detach().cpu().numpy())

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        train_acc = accuracy_score(answers, predictions)

        valid_loss, answers, predictions, prob = test_CNN(args, valid_loader, model, criterion, optimizer)

        valid_loss = np.mean(valid_loss)
        valid_acc = accuracy_score(answers, predictions)

        print("epoch: {}/{} | trn: {:.4f} / {:.4f} | val: {:.4f} / {:.4f}".format(
                    epoch, args.num_epoch, train_loss, train_acc, valid_loss, valid_acc
                ))
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        if (epoch==0) or (epoch>0 and (min(valid_loss_list[:-1])>valid_loss_list[-1])):
            torch.save({
                'epoch': epoch,
                'loss' : valid_loss_list[-1],
                'acc' : valid_acc_list[-1],
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'criterion' : criterion
            }, os.path.join(result_folder, f'cnn-swdr-classi-ecg-best.pt'))
            
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

    fig.suptitle("Loss", fontsize=30)
    fig.tight_layout()
    plt.savefig(os.path.join(result_folder, 'drcnn_loss.png'))
    pd.DataFrame([train_loss_list, valid_loss_list]).to_csv(os.path.join(result_folder, 'drcnn_loss.csv'), index=0)
    

def plot_confusion_matrix(y_label, y_pred, result_folder, normalized=True):
    mat = confusion_matrix(y_label, y_pred)
    cmn = mat.astype('float')
    if normalized:
        cmn = cmn / mat.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues')
    plt.ylabel('Actual', fontsize=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.title("Confusion Matrix (%)", fontsize=15)
    plt.savefig(os.path.join(result_folder,  f'drcnn_confmatrix.pdf'))
    
    
def inference_CNN(args, test_loader, model, criterion, optimizer, result_folder):
    print("Inference...")
    checkpoint = torch.load(os.path.join(result_folder, 'cnn-swdr-classi-ecg-best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = checkpoint['criterion']
    print(checkpoint['epoch'], checkpoint['loss'], checkpoint['acc'])
    
    model = model.eval()
    test_loss = []
    predictions = []
    answers = []
    prob = []
    with torch.no_grad():
        for (x, y) in tqdm(test_loader, leave=False):
            x = x.float().cuda()
            y = y.cuda()

            answers.extend(y.detach().cpu().numpy())


            outputs = model(x).squeeze(1)
            if args.model == 'EM':
                loss = criterion(outputs, y, r=1)
            else:
                loss = criterion(outputs, y) 

            prob.extend(torch.nn.Softmax()(outputs).detach().cpu().numpy())
            predictions.extend(torch.max(torch.nn.Softmax()(outputs),1)[1].detach().cpu().numpy())

            test_loss.append(loss.item())

    test_loss = np.mean(test_loss)
    acc = accuracy_score(answers, predictions)
    f1score = f1_score(answers, predictions, average='weighted')
    roc = roc_auc_score(answers, predictions)
    pr = precision_score(answers, predictions)
    
    plot_confusion_matrix(answers, predictions, result_folder)
    
    print (f'Performance: {acc} / {f1score} / {roc} / {pr}')
    
    result_label = pd.DataFrame([answers,predictions])
    result_pred = pd.DataFrame([predictions, prob])

    result_label.to_csv(os.path.join(result_folder,  f'cnndr_classification.csv'), index=0)
    result_pred.to_csv(os.path.join(result_folder,  f'cnndr_classification_prob.csv'), index=0)
    
    return acc, f1score, roc, pr