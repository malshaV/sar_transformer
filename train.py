# Code for training TransSAR on synthetic images 
# Author: Malsha Perera
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy




import argparse
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchvision import transforms as T
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init as init

from utils import BSD_SAR
from transform_main import TransSAR, TransSARV2, TransSARV3




parser = argparse.ArgumentParser(description='TransSAR')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run(default: 1)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 8)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lfw_path', default='../lfw', type=str, metavar='PATH',
                    help='path to root path of lfw dataset (default: ../lfw)')
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', required=True, type=str)
parser.add_argument('--modelname', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--save', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--model', default='TransSARV2', type=str,
                    help='model name')
parser.add_argument('--direc', required=True , type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int ,default=256)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--lambda_loss', default=0.04, type=float)

args = parser.parse_args()

aug = args.aug
direc = args.direc
num_epochs = args.epochs
modelname = args.modelname
crop_size = (args.crop, args.crop)
lambda_loss = args.lambda_loss

def total_variation(image_in):

    tv_h = torch.sum(torch.abs(image_in[ :, :-1] - image_in[ :, 1:]))
    tv_w = torch.sum(torch.abs(image_in[ :-1, :] - image_in[ 1:, :]))
    tv_loss = tv_h + tv_w

    return tv_loss 

def TV_loss(im_batch, weight):
    TV_L = 0.0

    for tv_idx in range(len(im_batch)):
        TV_L = TV_L + total_variation(im_batch[tv_idx,0,:,:])

    TV_L = TV_L/len(im_batch)

    return weight*TV_L

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)


train_dataset = BSD_SAR(args.train_dataset, crop_size, training_set=True)
val_dataset = BSD_SAR(args.val_dataset, crop_size, training_set=False)



dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device("cuda")


model = TransSARV2()

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)




pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


def train_model(model, criterion, optimizer, dataloader, valloader, direc, num_epochs=400):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                running_loss = 0.0
                running_loss_tv = 0.0
                for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):        
    
                    X_batch = Variable(X_batch.to(device ='cuda'))
                    y_batch = Variable(y_batch.to(device='cuda'))
                    
                    output = model(X_batch)

                    # print(output.size())

                
                    loss = criterion(output, y_batch)

                    
                    loss = loss + TV_loss(output,0.0000005)

                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    
                epoch_loss = running_loss / (batch_idx+1)
                
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                

                fulldir = direc+ "/all/" +"/{}/".format(epoch)
                
                if not os.path.isdir(fulldir):
                
                    os.makedirs(fulldir)
                torch.save(model.state_dict(), fulldir+args.model+".pth")

            else:
                model.eval()   # Set model to evaluate model
                running_loss = 0.0
                for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):

                    X_batch = Variable(X_batch.to(device='cuda'))
                    y_batch = Variable(y_batch.to(device='cuda'))

                    output = model(X_batch)

                    loss = criterion(output, y_batch)
                    

                    optimizer.zero_grad()

                    running_loss += loss.item()

                epoch_loss = running_loss / (batch_idx+1)
                print('{} Loss (MSE): {:.4f}'.format(phase, epoch_loss))
                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), direc+"model.pth")


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))


    model.load_state_dict(best_model_wts)
    return model



model_ft = train_model(model, criterion, optimizer, dataloader, valloader, direc, num_epochs)



                



    