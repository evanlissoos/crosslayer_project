import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import easydict
import os
from copy import deepcopy

args = easydict.EasyDict({
    "batch_size": 32,
    "epochs": 100,
    "lr": 0.001,
    "enable_cuda" : True,
    "L1norm" : False,
    "simpleNet" : True,
    "activation" : "relu", #relu, tanh, sigmoid
    "train_curve" : True, 
    "optimization" :"SGD",
    "cuda": False,
    "mps": False,
    "hooks": True
})
# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# MNIST Dataset (Images and Labels)
train_set = dsets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)
test_set = dsets.FashionMNIST(
    root = './data/FashionMNIST',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()                                 
    ])
)

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_set,
        batch_size = batch_size,
        shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_set,
        batch_size = batch_size,
        shuffle = False)

device = 'cpu'
if torch.cuda.is_available() and args.cuda:
    device = 'cuda'
elif torch.backends.mps.is_available() and args.mps:
    device = 'mps'

print('Using device: ' + str(device))

class MyConvNet(nn.Module):
    def __init__(self, args):
        super(MyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, 
                               padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.act1  = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, 
                               padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.act2  = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.lin2  = nn.Linear(7*7*32, 10)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        c1  = self.conv1(x)
        b1  = self.bn1(c1)
        a1  = self.act1(b1)
        p1  = self.pool1(a1)
        c2  = self.conv2(p1)
        b2  = self.bn2(c2)
        a2  = self.act2(b2)
        p2  = self.pool2(a2)
        flt = torch.flatten(p2, start_dim=1) # flt = p2.view(p2.size(0), -1)
        out = self.lin2(flt)
        out = self.dequant(out)
        return out

# Function that runs the testing set on the passed model
# Prints: accuracy of test set
# Returns: accuracy of test set
def test_model(model, en_print=True):
    criterion = nn.CrossEntropyLoss()
    criterion=criterion.to(device)
    model = model.to(device)
    correct = 0
    total = 0
    model.eval()
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        testloss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    if en_print:
        print('Accuracy for test images: % d %%' % (100 * correct / total))
    accuracy = 100*correct/total
    accuracy.to('cpu')
    return accuracy.item()

# Function that trains a model on the training set with set parameters
# Prints: progress of training
# Returns: trained model
def train_model(model, learning_rate=args.lr, num_epochs=args.epochs):
    criterion = nn.CrossEntropyLoss()
    criterion=criterion.to(device)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = Variable(labels).to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            L1norm = model.parameters()
            arr = []
            for name,param in model.named_parameters():
                if 'weight' in name.split('.'):
                    arr.append(param)
            L1loss = 0
            for Losstmp in arr:
                L1loss = L1loss+Losstmp.abs().mean()
            if(args.L1norm==True):
                if len(arr)>0:
                    loss = loss+L1loss/len(arr)

            loss.backward()
            optimizer.step()

            if (i + 1) % 600 == 0:
                print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                        % (epoch + 1, num_epochs, i + 1,
                        len(train_set) // batch_size, loss.data.item()))
    return model

