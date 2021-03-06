import torch
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from Se_ResNet import Se_ResNet
from torch import optim
from visdom import Visdom
import numpy as np

def train(model,data,target,loss_func,optimizer):
    optimizer.zero_grad()
    output=model(data)
    predictions = output.max(1,keepdim=True)[1]
    correct=predictions.eq(target.view_as(predictions)).sum().item()
    acc=correct / len(target)
    loss=loss_func(output,target)
    loss.backward()
    optimizer.step()
    return acc , loss

def test(model,test_loader,loss_func,use_cuda):
    acc_all=0
    loss_all=0
    step=0
    with torch.no_grad():
        for data ,target in test_loader:
            step += 1
            if use_cuda:
                data=data.cuda()
                target=target.cuda()
            output=model(data)
            predictions=output.max(1,keepdim=True)[1]
            correct=predictions.eq(target.view_as(predictions)).sum().item()
            acc=correct / len(target)
            loss=loss_func(output,target)
            acc_all += acc
            loss_all += loss
    return acc_all / step,loss_all / step
    
def adjust_lr(optimizer,epoch,lr):
    lr = lr * (0.1 ** int(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def main():
    viz = Visdom() 
    dataset_type='cifar100'
    
    num_epochs=100
    batch_size=64
    eval_step=1000
    base_lr = 0.01
    use_cuda=torch.cuda.is_available()
    
    dir_list = ('../data', '../data/MNIST', '../data/CIFAR-100')
    for directory in dir_list:
        if not os.path.exists(directory):
            os.mkdir(directory)
    if dataset_type=='mnist':
        train_loader=DataLoader(datasets.MNIST(root='../data/MNIST',train=True,download=True,transform=transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),batch_size=batch_size,shuffle=True)
        test_loader=DataLoader(datasets.MNIST(root='../data/MNIST',train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])),batch_size=batch_size)
    elif dataset_type=='cifar100':
        train_loader=DataLoader(datasets.CIFAR100(root='../data/CIFAR-100',train=True,download=True,transform=transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])),batch_size=batch_size,shuffle=True)
        test_loader=DataLoader(datasets.CIFAR100(root='../data/CIFAR-100',train=False,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])),batch_size=batch_size)
    else:
        raise ValueError('Wrong!')
        
    model = Se_ResNet()
    if use_cuda:
        model=model.cuda()
    ce_loss=torch.nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=1e-2,momentum=0.9,weight_decay=5e-4)
    train_step=0
    x,train_acc,test_acc = 0,0,0
    win = viz.line(
        X = np.array([x]),
        Y = np.column_stack((np.array([train_acc]),np.array([test_acc]))),
        opts = dict(
            title = "train ACC and test ACC",
            legend =["train_acc","test_acc"]
            )
        )
    for epoch in range(num_epochs):
        adjust_lr(optimizer,epoch,base_lr)
        for data, target in train_loader:
            train_step += 1
            if use_cuda:
                data=data.cuda()
                target=target.cuda()
            acc, loss=train(model,data,target,ce_loss,optimizer)
            if train_step % 100 ==0:
                print('Train set:Step:{},Loss:{:.4f},Accuracy:{:.2f}'.format(train_step,loss,acc))
                train_acc = acc
            if train_step % eval_step==0:
                acc, loss=test(model,test_loader,ce_loss,use_cuda)
                print('\nTest set: Step: {}, Loss: {:.4f},Accuracy: {:.2f}\n'.format(train_step,loss,acc))
                test_acc = acc
            if train_step % 100 ==0:
                viz.line(
                    X = np.array([train_step]),
                    Y = np.column_stack(
                        (np.array([train_acc]),np.array([test_acc])
                            )
                        ),
                    win = win,
                    update = "append"
                    )
if __name__ == '__main__':
    main()
