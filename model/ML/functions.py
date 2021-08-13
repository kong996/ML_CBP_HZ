from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchtext
import torchtext.vocab as Vocab
import numpy as np
import pandas as pd

class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
    
    
def train(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到
            
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = eva_acc(test_iter, net)
        
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
        
        train_ls.append(train_acc_sum / n)
        test_ls.append(test_acc)
    return train_ls, test_ls


def eva_acc(data_iter, net):
    device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) ==
                            y.to(device)).float().sum().cpu().item()
                
            else:
                print("the net is wrong")
            n += y.shape[0]
    return acc_sum / n 


def dnn_results(data_iter, net):
    device = list(net.parameters())[0].device
    i = 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                a = pd.DataFrame(X.numpy())
                df = pd.DataFrame(a, columns=[0,1,2,3,4,5,6,7])
                df.columns=['m1','m2','t1','t2','L1','L2','e_b','a_p']
                b = net(X.to(device)).argmax(dim=1).detach().numpy()
                df['predictions'] = b
                c = y.to(device)
                df['labels'] = c
                df.to_csv('df%d.csv'%i, index=False)
                i += 1
            else:
                print("the net is wrong")
    return i


def dnn_eva(data):
    p_s = data.predictions[data['labels']==1]
    p_u = data.predictions[data['labels']==0]
    TP, FN = p_s.value_counts() #设 TP>FN
    TN, FP = p_u.value_counts() #设 TN>FP
    acc = (TP+TN)/(TP+TN+FN+FP)
    precision = (TP/(TP+FP)) 
    recall = (TP/(TP+FN))
    F_1 = (2*precision*recall)/(precision+recall)
    print('Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F_1: %.4f'
          %(acc, precision, recall, F_1) )
    return TP, FN, FP, TN
    