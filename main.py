import os
import numpy as np

import torch
from torchvision.transforms import transforms as T
import argparse 
import torch.nn.functional as F

from torch import optim
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader


from model import UNet_2d,UNet_3d
from loss import DiceLoss,dice_coeff
from data import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cuda:1")


x_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5,0.5], [0.5, 0.5, 0.5,0.5])
])


x_transform = T.ToTensor()
y_transform = T.ToTensor()


def train_model(model,criterion,optimizer,dataload,num_epochs):
    for epoch in range(num_epochs):
        print('-' * 30)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            optimizer.zero_grad()
            inputs = x.type(torch.FloatTensor).to(device)
            labels = y.type(torch.FloatTensor).to(device)
            outputs = model(inputs)
            dice_loss = criterion[0](outputs, labels)
            bce_loss = criterion[1](outputs, labels)
            loss = dice_loss + bce_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            #print("%d/%d,train_loss:%0.4f" % (step, dataset_size // dataload.batch_size, loss.item()))
        epoch_loss /= step
        print("epoch %d loss:%0.4f" % (epoch, epoch_loss))
    torch.save(model.state_dict(),'unet_spleen_%d.pth' % epoch)
    return model


def train():

    model = UNet_2d(8,1,1).to(device) # conv_channels=8, input_channels, classes, slices
    print(model)
    
    criterion = [DiceLoss(),torch.nn.BCELoss()]
        
    optimizer = optim.Adam(model.parameters(), lr = args.lr, eps=1e-8) #optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)#
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, cooldown=0, min_lr=1e-8)
    
    dataset = SpleenDataset(transform=x_transform, target_transform=y_transform)#
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,num_workers=4)
    train_model(model,criterion,optimizer,dataloader,args.num_epochs)



def test():
    model = UNet_2d(8,1,1).to(device)
    model.load_state_dict(torch.load(args.weight))
    model.eval()
    
    dataset = SpleenDataset(transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(dataset)
    #import matplotlib.pyplot as plt
    #plt.ion()
    dice = 0
    num = 0
    with torch.no_grad():
        for x, _ in dataloaders:
            inputs = x.type(torch.FloatTensor).to(device)
            labels = _.type(torch.FloatTensor).to(device)
            #x = torch.tensor(x, dtype=torch.float32)
            #_ = torch.tensor(_, dtype=torch.float32)
            y = model(inputs)
            dice += dice_coeff(y,labels)
            num += 1
            
            print(dice)
            #img_y=torch.squeeze(y).numpy()
            #plt.imshow(img_y)
            #plt.pause(0.01)
        #plt.show()
        dice /= num
        print('Dice :', dice)



if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('action', type=str, help='train or test')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight', type=str, help='the path of the mode weight file')
    args = parser.parse_args()
    
    if args.action == 'train':
        train()
    if args.action == 'test':
        test()
