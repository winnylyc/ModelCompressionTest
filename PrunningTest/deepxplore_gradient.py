import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time
import sys
import torch.quantization
from utils import progress_bar
import torch.nn.utils.prune as prune
import random
import os 
import torch.utils.data as data
from glob import glob
from torch.autograd import Variable
import torch.optim as optim
from copy import deepcopy
import gc

import cv2 
from collections import OrderedDict

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG_no_sequntial_relu(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_no_sequntial_relu, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
        )
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
        )
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
        )
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
        )
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
        )
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(kernel_size=1, stride=1) 
        self.relu = nn.ReLU(inplace=False)      
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.max3(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.max4(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.relu(x)
        x = self.max5(x)
        out = self.avg(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class VGG_no_sequntial(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_no_sequntial, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(kernel_size=1, stride=1)       
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max3(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.max5(x)
        out = self.avg(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def load_model(model_file):
    # model = VGG_no_sequntial_relu('VGG11')
    # model = ResNet18()
    # model = GoogLeNet()
    # model = GoogLeNet_relu()
    # model = DenseNet121()
    model = VGG('VGG16')
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['net'])
    model.to(device)
    return model

def load_model_2(model_file):
    # model = VGG_no_sequntial_relu('VGG11')
    model = ResNet18()
    state_dict = torch.load(model_file)
    new_state_dict = OrderedDict()
    for k, v in state_dict['state_dict'].items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    return model

def test(epoch, net, testloader):
    global best_acc
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx < 50:
                continue
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return correct/total

def save_tensor(image_numpy, batch_idx, p, root_path, ori_target):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    # image_numpy = inputs[0].cpu().detach().float().numpy()

    for i in range(len(mean)):
        image_numpy[i] = image_numpy[i] * std[i] + mean[i]
    
    image_numpy = image_numpy * 255
    image_numpy = np.transpose(image_numpy, (1, 2, 0)).astype(np.uint8)
    if p >= 0:
        cv2.imwrite(root_path + str(batch_idx) + '_' + str(p) + '_' + str(ori_target) + '.jpg', image_numpy)
    else:
        cv2.imwrite(root_path + str(batch_idx) + '.jpg', image_numpy)
    
    # print(batch_idx)

class Dataset_Diff(data.Dataset):
    def __init__(self, path):
        self.frames = []
        self.setup(path)
    
    def setup(self, path):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        for path in glob(path + '/*.jpg'):
            # print(path)
            self.frames.append(transform_test(cv2.imread(path)))

    
    def __getitem__(self, index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)

activation = []
# def get_activation(name):
def hook(model, input, output):
    # activation.append(output.detach())
    activation.append(output)
    # return hook

activation_ori = []
# def get_activation_ori(name):
def hook_ori(model, input, output):
    # activation.append(output.detach())
    activation_ori.append(output)
    # return hook_ori

def main():
    print('==> Preparing data..')
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    np.random.seed(2)
    random.seed(2)  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='../pytorch-cifar/data', train=False, download=True, transform=transform_test)
    testloader_test = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)
    
    trainset = torchvision.datasets.CIFAR10(
        root='../pytorch-cifar/data', train=True, download=True, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    # myModel = load_model('../pytorch-cifar/checkpoint_relu_1113/ckpt.pth').to(device)
    # myModel = load_model('../pytorch-cifar/checkpoint_resnet18/ckpt.pth').to(device)
    # myModel = load_model('../pytorch-cifar/checkpoint_googlenet_relu/ckpt.pth').to(device)
    # myModel = load_model('../pytorch-cifar/checkpoint_densenet/ckpt.pth').to(device)
    myModel = load_model('../pytorch-cifar/checkpoint_vgg16/ckpt.pth').to(device)
    print('success')
    print(myModel.state_dict().keys())
    print(myModel)
    "vgg"
    # parameters_to_prune = (
    #     (myModel.conv1[0], 'weight'),
    #     (myModel.conv2[0], 'weight'),
    #     (myModel.conv3[0], 'weight'),
    #     (myModel.conv4[0], 'weight'),
    #     (myModel.conv5[0], 'weight'),
    #     (myModel.conv6[0], 'weight'),
    #     (myModel.conv7[0], 'weight'),
    #     (myModel.conv8[0], 'weight'),
    # )
    "vgg16"
    parameters_to_prune = (
        (myModel.features[0], 'weight'),
        (myModel.features[3], 'weight'),
        (myModel.features[7], 'weight'),
        (myModel.features[10], 'weight'),
        (myModel.features[14], 'weight'),
        (myModel.features[17], 'weight'),
        (myModel.features[20], 'weight'),
        (myModel.features[27], 'weight'),
        (myModel.features[30], 'weight'),
        (myModel.features[34], 'weight'),
        (myModel.features[37], 'weight'),
        (myModel.features[40], 'weight'),
    )
    "resnet"
    # parameters_to_prune = (
    #     (myModel.conv1, 'weight'),
    #     (myModel.layer1[0].conv1, 'weight'),
    #     (myModel.layer1[0].conv2, 'weight'),
    #     (myModel.layer1[1].conv1, 'weight'),
    #     (myModel.layer1[1].conv2, 'weight'),
    #     (myModel.layer2[0].conv1, 'weight'),
    #     (myModel.layer2[0].conv2, 'weight'),
    #     (myModel.layer2[0].shortcut[0], 'weight'),
    #     (myModel.layer2[1].conv1, 'weight'),
    #     (myModel.layer2[1].conv2, 'weight'),
    #     (myModel.layer3[0].conv1, 'weight'),
    #     (myModel.layer3[0].conv2, 'weight'),
    #     (myModel.layer3[0].shortcut[0], 'weight'),
    #     (myModel.layer3[1].conv1, 'weight'),
    #     (myModel.layer3[1].conv2, 'weight'),
    #     (myModel.layer4[0].conv1, 'weight'),
    #     (myModel.layer4[0].conv2, 'weight'),
    #     (myModel.layer4[0].shortcut[0], 'weight'),
    #     (myModel.layer4[1].conv1, 'weight'),
    #     (myModel.layer4[1].conv2, 'weight'),
    # )
    "googlenet"
    # parameters_to_prune = {
    #     (myModel.pre_layers[0], 'weight'),
    #     (myModel.a3.b1[0], 'weight'),
    #     (myModel.a3.b2[0], 'weight'),
    #     (myModel.a3.b2[3], 'weight'),
    #     (myModel.a3.b3[0], 'weight'),
    #     (myModel.a3.b3[3], 'weight'),
    #     (myModel.a3.b3[6], 'weight'),
    #     (myModel.a3.b4[1], 'weight'),
    #     (myModel.b3.b1[0], 'weight'),
    #     (myModel.b3.b2[0], 'weight'),
    #     (myModel.b3.b2[3], 'weight'),
    #     (myModel.b3.b3[0], 'weight'),
    #     (myModel.b3.b3[3], 'weight'),
    #     (myModel.b3.b3[6], 'weight'),
    #     (myModel.b3.b4[1], 'weight'),
    #     (myModel.a4.b1[0], 'weight'),
    #     (myModel.a4.b2[0], 'weight'),
    #     (myModel.a4.b2[3], 'weight'),
    #     (myModel.a4.b3[0], 'weight'),
    #     (myModel.a4.b3[3], 'weight'),
    #     (myModel.a4.b3[6], 'weight'),
    #     (myModel.a4.b4[1], 'weight'),
    #     (myModel.b4.b1[0], 'weight'),
    #     (myModel.b4.b2[0], 'weight'),
    #     (myModel.b4.b2[3], 'weight'),
    #     (myModel.c4.b3[0], 'weight'),
    #     (myModel.c4.b3[3], 'weight'),
    #     (myModel.c4.b3[6], 'weight'),
    #     (myModel.c4.b4[1], 'weight'),
    #     (myModel.d4.b1[0], 'weight'),
    #     (myModel.d4.b2[0], 'weight'),
    #     (myModel.d4.b2[3], 'weight'),
    #     (myModel.d4.b3[0], 'weight'),
    #     (myModel.d4.b3[3], 'weight'),
    #     (myModel.d4.b3[6], 'weight'),
    #     (myModel.d4.b4[1], 'weight'),
    #     (myModel.e4.b1[0], 'weight'),
    #     (myModel.e4.b2[0], 'weight'),
    #     (myModel.e4.b2[3], 'weight'),
    #     (myModel.e4.b3[0], 'weight'),
    #     (myModel.e4.b3[3], 'weight'),
    #     (myModel.e4.b3[6], 'weight'),
    #     (myModel.b4.b3[0], 'weight'),
    #     (myModel.b4.b3[3], 'weight'),
    #     (myModel.b4.b3[6], 'weight'),
    #     (myModel.b4.b4[1], 'weight'),
    #     (myModel.c4.b1[0], 'weight'),
    #     (myModel.c4.b2[0], 'weight'),
    #     (myModel.c4.b2[3], 'weight'),
    #     (myModel.e4.b4[1], 'weight'),
    #     (myModel.a5.b1[0], 'weight'),
    #     (myModel.a5.b2[0], 'weight'),
    #     (myModel.a5.b2[3], 'weight'),
    #     (myModel.a5.b3[0], 'weight'),
    #     (myModel.a5.b3[3], 'weight'),
    #     (myModel.a5.b3[6], 'weight'),
    #     (myModel.a5.b4[1], 'weight'),
    #     (myModel.b5.b1[0], 'weight'),
    #     (myModel.b5.b2[0], 'weight'),
    #     (myModel.b5.b2[3], 'weight'),
    #     (myModel.b5.b3[0], 'weight'),
    #     (myModel.b5.b3[3], 'weight'),
    #     (myModel.b5.b3[6], 'weight'),
    #     (myModel.b5.b4[1], 'weight'),
    # }
    "densenet"
    # parameters_to_prune = {
    #     (myModel.conv1, 'weight'),
    #     (myModel.dense1[0].conv1, 'weight'),
    #     (myModel.dense1[0].conv2, 'weight'),
    #     (myModel.dense1[1].conv1, 'weight'),
    #     (myModel.dense1[1].conv2, 'weight'),
    #     (myModel.dense1[2].conv1, 'weight'),
    #     (myModel.dense1[2].conv2, 'weight'),
    #     (myModel.dense1[3].conv1, 'weight'),
    #     (myModel.dense1[3].conv2, 'weight'),
    #     (myModel.dense1[4].conv1, 'weight'),
    #     (myModel.dense1[4].conv2, 'weight'),
    #     (myModel.dense1[5].conv1, 'weight'),
    #     (myModel.dense1[5].conv2, 'weight'),
    #     (myModel.trans1.conv, 'weight'),
    #     (myModel.dense2[0].conv1, 'weight'),
    #     (myModel.dense2[0].conv2, 'weight'),
    #     (myModel.dense2[1].conv1, 'weight'),
    #     (myModel.dense2[1].conv2, 'weight'),
    #     (myModel.dense2[2].conv1, 'weight'),
    #     (myModel.dense2[2].conv2, 'weight'),
    #     (myModel.dense2[3].conv1, 'weight'),
    #     (myModel.dense2[3].conv2, 'weight'),
    #     (myModel.dense2[4].conv1, 'weight'),
    #     (myModel.dense2[4].conv2, 'weight'),
    #     (myModel.dense2[5].conv1, 'weight'),
    #     (myModel.dense2[5].conv2, 'weight'),
    #     (myModel.dense2[6].conv1, 'weight'),
    #     (myModel.dense2[6].conv2, 'weight'),
    #     (myModel.dense2[7].conv1, 'weight'),
    #     (myModel.dense2[7].conv2, 'weight'),
    #     (myModel.dense2[8].conv1, 'weight'),
    #     (myModel.dense2[8].conv2, 'weight'),
    #     (myModel.dense2[9].conv1, 'weight'),
    #     (myModel.dense2[9].conv2, 'weight'),
    #     (myModel.dense2[10].conv1, 'weight'),
    #     (myModel.dense2[10].conv2, 'weight'),
    #     (myModel.dense2[11].conv1, 'weight'),
    #     (myModel.dense2[11].conv2, 'weight'),
    #     (myModel.trans2.conv, 'weight'),
    #     (myModel.dense3[0].conv1, 'weight'),
    #     (myModel.dense3[0].conv2, 'weight'),
    #     (myModel.dense3[1].conv1, 'weight'),
    #     (myModel.dense3[1].conv2, 'weight'),
    #     (myModel.dense3[2].conv1, 'weight'),
    #     (myModel.dense3[2].conv2, 'weight'),
    #     (myModel.dense3[3].conv1, 'weight'),
    #     (myModel.dense3[3].conv2, 'weight'),
    #     (myModel.dense3[4].conv1, 'weight'),
    #     (myModel.dense3[4].conv2, 'weight'),
    #     (myModel.dense3[5].conv1, 'weight'),
    #     (myModel.dense3[5].conv2, 'weight'),
    #     (myModel.dense3[6].conv1, 'weight'),
    #     (myModel.dense3[6].conv2, 'weight'),
    #     (myModel.dense3[7].conv1, 'weight'),
    #     (myModel.dense3[7].conv2, 'weight'),
    #     (myModel.dense3[8].conv1, 'weight'),
    #     (myModel.dense3[8].conv2, 'weight'),
    #     (myModel.dense3[9].conv1, 'weight'),
    #     (myModel.dense3[9].conv2, 'weight'),
    #     (myModel.dense3[10].conv1, 'weight'),
    #     (myModel.dense3[10].conv2, 'weight'),
    #     (myModel.dense3[11].conv1, 'weight'),
    #     (myModel.dense3[11].conv2, 'weight'),
    #     (myModel.dense3[12].conv1, 'weight'),
    #     (myModel.dense3[12].conv2, 'weight'),
    #     (myModel.dense3[13].conv1, 'weight'),
    #     (myModel.dense3[13].conv2, 'weight'),
    #     (myModel.dense3[14].conv1, 'weight'),
    #     (myModel.dense3[14].conv2, 'weight'),
    #     (myModel.dense3[15].conv1, 'weight'),
    #     (myModel.dense3[15].conv2, 'weight'),
    #     (myModel.dense3[16].conv1, 'weight'),
    #     (myModel.dense3[16].conv2, 'weight'),
    #     (myModel.dense3[17].conv1, 'weight'),
    #     (myModel.dense3[17].conv2, 'weight'),
    #     (myModel.dense3[18].conv1, 'weight'),
    #     (myModel.dense3[18].conv2, 'weight'),
    #     (myModel.dense3[19].conv1, 'weight'),
    #     (myModel.dense3[19].conv2, 'weight'),
    #     (myModel.dense3[20].conv1, 'weight'),
    #     (myModel.dense3[20].conv2, 'weight'),
    #     (myModel.dense3[21].conv1, 'weight'),
    #     (myModel.dense3[21].conv2, 'weight'),
    #     (myModel.dense3[22].conv1, 'weight'),
    #     (myModel.dense3[22].conv2, 'weight'),
    #     (myModel.dense3[23].conv1, 'weight'),
    #     (myModel.dense3[23].conv2, 'weight'),
    #     (myModel.trans3.conv, 'weight'),
    #     (myModel.dense4[0].conv1, 'weight'),
    #     (myModel.dense4[0].conv2, 'weight'),
    #     (myModel.dense4[1].conv1, 'weight'),
    #     (myModel.dense4[1].conv2, 'weight'),
    #     (myModel.dense4[2].conv1, 'weight'),
    #     (myModel.dense4[2].conv2, 'weight'),
    #     (myModel.dense4[3].conv1, 'weight'),
    #     (myModel.dense4[3].conv2, 'weight'),
    #     (myModel.dense4[4].conv1, 'weight'),
    #     (myModel.dense4[4].conv2, 'weight'),
    #     (myModel.dense4[5].conv1, 'weight'),
    #     (myModel.dense4[5].conv2, 'weight'),
    #     (myModel.dense4[6].conv1, 'weight'),
    #     (myModel.dense4[6].conv2, 'weight'),
    #     (myModel.dense4[7].conv1, 'weight'),
    #     (myModel.dense4[7].conv2, 'weight'),
    #     (myModel.dense4[8].conv1, 'weight'),
    #     (myModel.dense4[8].conv2, 'weight'),
    #     (myModel.dense4[9].conv1, 'weight'),
    #     (myModel.dense4[9].conv2, 'weight'),
    #     (myModel.dense4[10].conv1, 'weight'),
    #     (myModel.dense4[10].conv2, 'weight'),
    #     (myModel.dense4[11].conv1, 'weight'),
    #     (myModel.dense4[11].conv2, 'weight'),
    #     (myModel.dense4[12].conv1, 'weight'),
    #     (myModel.dense4[12].conv2, 'weight'),
    #     (myModel.dense4[13].conv1, 'weight'),
    #     (myModel.dense4[13].conv2, 'weight'),
    #     (myModel.dense4[14].conv1, 'weight'),
    #     (myModel.dense4[14].conv2, 'weight'),
    #     (myModel.dense4[15].conv1, 'weight'),
    #     (myModel.dense4[15].conv2, 'weight'),
    # }
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.75,
    )
    print('success')
    # state_dict = torch.load('2checkpoint_vgg_finetune_rate75/ckpt.pth')
    # state_dict = torch.load('2checkpoint_resnet_finetune_rate75/ckpt.pth')
    # state_dict = torch.load('2checkpoint_google_finetune_rate75/ckpt.pth')
    state_dict = torch.load('2checkpoint_vgg16_finetune_rate75/ckpt.pth')
    myModel.load_state_dict(state_dict['net'])
    # oriModel = load_model('../pytorch-cifar/checkpoint_relu_1113/ckpt.pth').to(device)
    # oriModel = load_model('../pytorch-cifar/checkpoint_resnet18/ckpt.pth').to(device)
    # oriModel = load_model('../pytorch-cifar/checkpoint_googlenet_relu/ckpt.pth').to(device)
    oriModel = load_model('../pytorch-cifar/checkpoint_vgg16/ckpt.pth').to(device)
    myModel.eval()
    oriModel.eval()
    acc_p = test(0, myModel, testloader_test)
    acc_o = test(0, oriModel, testloader_test)
    # print(myModel.state_dict().keys())
    # for parameters in myModel.parameters():
    #     print(parameters)
    # print(oriModel.state_dict().keys())
    # print(oriModel)
    # print(myModel.conv1[0].state_dict()[])
    # for parameters in oriModel.parameters():
    #     print(parameters)
    # test(0, myModel, testloader)
    # test(0, oriModel, testloader)
    root_path2 = '2PruningDiffDeepGra_vgg16_rate75_test/'
    os.makedirs(root_path2, exist_ok = True)
    var = [0.0608027,  0.05892733, 0.06850188]
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    loss_fn_AE_ce = nn.CrossEntropyLoss(reduction = 'none')
    count_ori_error = 0
    criterion = nn.CrossEntropyLoss()
    "vgg"
    # myModel.conv1.register_forward_hook(hook)
    # myModel.conv2.register_forward_hook(hook)
    # myModel.conv3.register_forward_hook(hook)
    # myModel.conv4.register_forward_hook(hook)
    # myModel.conv5.register_forward_hook(hook)
    # myModel.conv6.register_forward_hook(hook)
    # myModel.conv7.register_forward_hook(hook)
    # myModel.conv8.register_forward_hook(hook)
    # oriModel.conv1.register_forward_hook(hook_ori)
    # oriModel.conv2.register_forward_hook(hook_ori)
    # oriModel.conv3.register_forward_hook(hook_ori)
    # oriModel.conv4.register_forward_hook(hook_ori)
    # oriModel.conv5.register_forward_hook(hook_ori)
    # oriModel.conv6.register_forward_hook(hook_ori)
    # oriModel.conv7.register_forward_hook(hook_ori)
    # oriModel.conv8.register_forward_hook(hook_ori)
    "resnet"
    # myModel.conv1.register_forward_hook(hook)
    # myModel.layer1[0].conv1.register_forward_hook(hook)
    # myModel.layer1[0].conv2.register_forward_hook(hook)
    # myModel.layer1[1].conv1.register_forward_hook(hook)
    # myModel.layer1[1].conv2.register_forward_hook(hook)
    # myModel.layer2[0].conv1.register_forward_hook(hook)
    # myModel.layer2[0].conv2.register_forward_hook(hook)
    # myModel.layer2[0].shortcut[0].register_forward_hook(hook)
    # myModel.layer2[1].conv1.register_forward_hook(hook)
    # myModel.layer2[1].conv2.register_forward_hook(hook)
    # myModel.layer3[0].conv1.register_forward_hook(hook)
    # myModel.layer3[0].conv2.register_forward_hook(hook)
    # myModel.layer3[0].shortcut[0].register_forward_hook(hook)
    # myModel.layer3[1].conv1.register_forward_hook(hook)
    # myModel.layer3[1].conv2.register_forward_hook(hook)
    # myModel.layer4[0].conv1.register_forward_hook(hook)
    # myModel.layer4[0].conv2.register_forward_hook(hook)
    # myModel.layer4[0].shortcut[0].register_forward_hook(hook)
    # myModel.layer4[1].conv1.register_forward_hook(hook)
    # myModel.layer4[1].conv2.register_forward_hook(hook)
    # oriModel.conv1.register_forward_hook(hook_ori)
    # oriModel.layer1[0].conv1.register_forward_hook(hook_ori)
    # oriModel.layer1[0].conv2.register_forward_hook(hook_ori)
    # oriModel.layer1[1].conv1.register_forward_hook(hook_ori)
    # oriModel.layer1[1].conv2.register_forward_hook(hook_ori)
    # oriModel.layer2[0].conv1.register_forward_hook(hook_ori)
    # oriModel.layer2[0].conv2.register_forward_hook(hook_ori)
    # oriModel.layer2[0].shortcut[0].register_forward_hook(hook_ori)
    # oriModel.layer2[1].conv1.register_forward_hook(hook_ori)
    # oriModel.layer2[1].conv2.register_forward_hook(hook_ori)
    # oriModel.layer3[0].conv1.register_forward_hook(hook_ori)
    # oriModel.layer3[0].conv2.register_forward_hook(hook_ori)
    # oriModel.layer3[0].shortcut[0].register_forward_hook(hook_ori)
    # oriModel.layer3[1].conv1.register_forward_hook(hook_ori)
    # oriModel.layer3[1].conv2.register_forward_hook(hook_ori)
    # oriModel.layer4[0].conv1.register_forward_hook(hook_ori)
    # oriModel.layer4[0].conv2.register_forward_hook(hook_ori)
    # oriModel.layer4[0].shortcut[0].register_forward_hook(hook_ori)
    # oriModel.layer4[1].conv1.register_forward_hook(hook_ori)
    # oriModel.layer4[1].conv2.register_forward_hook(hook_ori)
    "googlenet"
    # myModel.pre_layers[0].register_forward_hook(hook)
    # myModel.a3.b1[0].register_forward_hook(hook)
    # myModel.a3.b2[0].register_forward_hook(hook)
    # myModel.a3.b2[3].register_forward_hook(hook)
    # myModel.a3.b3[0].register_forward_hook(hook)
    # myModel.a3.b3[3].register_forward_hook(hook)
    # myModel.a3.b3[6].register_forward_hook(hook)
    # myModel.a3.b4[1].register_forward_hook(hook)
    # myModel.b3.b1[0].register_forward_hook(hook)
    # myModel.b3.b2[0].register_forward_hook(hook)
    # myModel.b3.b2[3].register_forward_hook(hook)
    # myModel.b3.b3[0].register_forward_hook(hook)
    # myModel.b3.b3[3].register_forward_hook(hook)
    # myModel.b3.b3[6].register_forward_hook(hook)
    # myModel.b3.b4[1].register_forward_hook(hook)
    # myModel.a4.b1[0].register_forward_hook(hook)
    # myModel.a4.b2[0].register_forward_hook(hook)
    # myModel.a4.b2[3].register_forward_hook(hook)
    # myModel.a4.b3[0].register_forward_hook(hook)
    # myModel.a4.b3[3].register_forward_hook(hook)
    # myModel.a4.b3[6].register_forward_hook(hook)
    # myModel.a4.b4[1].register_forward_hook(hook)
    # myModel.b4.b1[0].register_forward_hook(hook)
    # myModel.b4.b2[0].register_forward_hook(hook)
    # myModel.b4.b2[3].register_forward_hook(hook)
    # myModel.b4.b3[0].register_forward_hook(hook)
    # myModel.b4.b3[3].register_forward_hook(hook)
    # myModel.b4.b3[6].register_forward_hook(hook)
    # myModel.b4.b4[1].register_forward_hook(hook)
    # myModel.c4.b1[0].register_forward_hook(hook)
    # myModel.c4.b2[0].register_forward_hook(hook)
    # myModel.c4.b2[3].register_forward_hook(hook)
    # myModel.c4.b3[0].register_forward_hook(hook)
    # myModel.c4.b3[3].register_forward_hook(hook)
    # myModel.c4.b3[6].register_forward_hook(hook)
    # myModel.c4.b4[1].register_forward_hook(hook)
    # myModel.d4.b1[0].register_forward_hook(hook)
    # myModel.d4.b2[0].register_forward_hook(hook)
    # myModel.d4.b2[3].register_forward_hook(hook)
    # myModel.d4.b3[0].register_forward_hook(hook)
    # myModel.d4.b3[3].register_forward_hook(hook)
    # myModel.d4.b3[6].register_forward_hook(hook)
    # myModel.d4.b4[1].register_forward_hook(hook)
    # myModel.e4.b1[0].register_forward_hook(hook)
    # myModel.e4.b2[0].register_forward_hook(hook)
    # myModel.e4.b2[3].register_forward_hook(hook)
    # myModel.e4.b3[0].register_forward_hook(hook)
    # myModel.e4.b3[3].register_forward_hook(hook)
    # myModel.e4.b3[6].register_forward_hook(hook)
    # myModel.e4.b4[1].register_forward_hook(hook)
    # myModel.a5.b1[0].register_forward_hook(hook)
    # myModel.a5.b2[0].register_forward_hook(hook)
    # myModel.a5.b2[3].register_forward_hook(hook)
    # myModel.a5.b3[0].register_forward_hook(hook)
    # myModel.a5.b3[3].register_forward_hook(hook)
    # myModel.a5.b3[6].register_forward_hook(hook)
    # myModel.a5.b4[1].register_forward_hook(hook)
    # myModel.b5.b1[0].register_forward_hook(hook)
    # myModel.b5.b2[0].register_forward_hook(hook)
    # myModel.b5.b2[3].register_forward_hook(hook)
    # myModel.b5.b3[0].register_forward_hook(hook)
    # myModel.b5.b3[3].register_forward_hook(hook)
    # myModel.b5.b3[6].register_forward_hook(hook)
    # myModel.b5.b4[1].register_forward_hook(hook)
    # oriModel.pre_layers[0].register_forward_hook(hook_ori)
    # oriModel.a3.b1[0].register_forward_hook(hook_ori)
    # oriModel.a3.b2[0].register_forward_hook(hook_ori)
    # oriModel.a3.b2[3].register_forward_hook(hook_ori)
    # oriModel.a3.b3[0].register_forward_hook(hook_ori)
    # oriModel.a3.b3[3].register_forward_hook(hook_ori)
    # oriModel.a3.b3[6].register_forward_hook(hook_ori)
    # oriModel.a3.b4[1].register_forward_hook(hook_ori)
    # oriModel.b3.b1[0].register_forward_hook(hook_ori)
    # oriModel.b3.b2[0].register_forward_hook(hook_ori)
    # oriModel.b3.b2[3].register_forward_hook(hook_ori)
    # oriModel.b3.b3[0].register_forward_hook(hook_ori)
    # oriModel.b3.b3[3].register_forward_hook(hook_ori)
    # oriModel.b3.b3[6].register_forward_hook(hook_ori)
    # oriModel.b3.b4[1].register_forward_hook(hook_ori)
    # oriModel.a4.b1[0].register_forward_hook(hook_ori)
    # oriModel.a4.b2[0].register_forward_hook(hook_ori)
    # oriModel.a4.b2[3].register_forward_hook(hook_ori)
    # oriModel.a4.b3[0].register_forward_hook(hook_ori)
    # oriModel.a4.b3[3].register_forward_hook(hook_ori)
    # oriModel.a4.b3[6].register_forward_hook(hook_ori)
    # oriModel.a4.b4[1].register_forward_hook(hook_ori)
    # oriModel.b4.b1[0].register_forward_hook(hook_ori)
    # oriModel.b4.b2[0].register_forward_hook(hook_ori)
    # oriModel.b4.b2[3].register_forward_hook(hook_ori)
    # oriModel.b4.b3[0].register_forward_hook(hook_ori)
    # oriModel.b4.b3[3].register_forward_hook(hook_ori)
    # oriModel.b4.b3[6].register_forward_hook(hook_ori)
    # oriModel.b4.b4[1].register_forward_hook(hook_ori)
    # oriModel.c4.b1[0].register_forward_hook(hook_ori)
    # oriModel.c4.b2[0].register_forward_hook(hook_ori)
    # oriModel.c4.b2[3].register_forward_hook(hook_ori)
    # oriModel.c4.b3[0].register_forward_hook(hook_ori)
    # oriModel.c4.b3[3].register_forward_hook(hook_ori)
    # oriModel.c4.b3[6].register_forward_hook(hook_ori)
    # oriModel.c4.b4[1].register_forward_hook(hook_ori)
    # oriModel.d4.b1[0].register_forward_hook(hook_ori)
    # oriModel.d4.b2[0].register_forward_hook(hook_ori)
    # oriModel.d4.b2[3].register_forward_hook(hook_ori)
    # oriModel.d4.b3[0].register_forward_hook(hook_ori)
    # oriModel.d4.b3[3].register_forward_hook(hook_ori)
    # oriModel.d4.b3[6].register_forward_hook(hook_ori)
    # oriModel.d4.b4[1].register_forward_hook(hook_ori)
    # oriModel.e4.b1[0].register_forward_hook(hook_ori)
    # oriModel.e4.b2[0].register_forward_hook(hook_ori)
    # oriModel.e4.b2[3].register_forward_hook(hook_ori)
    # oriModel.e4.b3[0].register_forward_hook(hook_ori)
    # oriModel.e4.b3[3].register_forward_hook(hook_ori)
    # oriModel.e4.b3[6].register_forward_hook(hook_ori)
    # oriModel.e4.b4[1].register_forward_hook(hook_ori)
    # oriModel.a5.b1[0].register_forward_hook(hook_ori)
    # oriModel.a5.b2[0].register_forward_hook(hook_ori)
    # oriModel.a5.b2[3].register_forward_hook(hook_ori)
    # oriModel.a5.b3[0].register_forward_hook(hook_ori)
    # oriModel.a5.b3[3].register_forward_hook(hook_ori)
    # oriModel.a5.b3[6].register_forward_hook(hook_ori)
    # oriModel.a5.b4[1].register_forward_hook(hook_ori)
    # oriModel.b5.b1[0].register_forward_hook(hook_ori)
    # oriModel.b5.b2[0].register_forward_hook(hook_ori)
    # oriModel.b5.b2[3].register_forward_hook(hook_ori)
    # oriModel.b5.b3[0].register_forward_hook(hook_ori)
    # oriModel.b5.b3[3].register_forward_hook(hook_ori)
    # oriModel.b5.b3[6].register_forward_hook(hook_ori)
    # oriModel.b5.b4[1].register_forward_hook(hook_ori)

    # myModel.pre_layers[1].register_forward_hook(hook)
    # myModel.a3.b1[1].register_forward_hook(hook)
    # myModel.a3.b2[1].register_forward_hook(hook)
    # myModel.a3.b2[4].register_forward_hook(hook)
    # myModel.a3.b3[1].register_forward_hook(hook)
    # myModel.a3.b3[4].register_forward_hook(hook)
    # myModel.a3.b3[7].register_forward_hook(hook)
    # myModel.a3.b4[2].register_forward_hook(hook)
    # myModel.b3.b1[1].register_forward_hook(hook)
    # myModel.b3.b2[1].register_forward_hook(hook)
    # myModel.b3.b2[4].register_forward_hook(hook)
    # myModel.b3.b3[1].register_forward_hook(hook)
    # myModel.b3.b3[4].register_forward_hook(hook)
    # myModel.b3.b3[7].register_forward_hook(hook)
    # myModel.b3.b4[2].register_forward_hook(hook)
    # myModel.a4.b1[1].register_forward_hook(hook)
    # myModel.a4.b2[1].register_forward_hook(hook)
    # myModel.a4.b2[4].register_forward_hook(hook)
    # myModel.a4.b3[1].register_forward_hook(hook)
    # myModel.a4.b3[4].register_forward_hook(hook)
    # myModel.a4.b3[7].register_forward_hook(hook)
    # myModel.a4.b4[2].register_forward_hook(hook)
    # myModel.b4.b1[1].register_forward_hook(hook)
    # myModel.b4.b2[1].register_forward_hook(hook)
    # myModel.b4.b2[4].register_forward_hook(hook)
    # myModel.b4.b3[1].register_forward_hook(hook)
    # myModel.b4.b3[4].register_forward_hook(hook)
    # myModel.b4.b3[7].register_forward_hook(hook)
    # myModel.b4.b4[2].register_forward_hook(hook)
    # myModel.c4.b1[1].register_forward_hook(hook)
    # myModel.c4.b2[1].register_forward_hook(hook)
    # myModel.c4.b2[4].register_forward_hook(hook)
    # myModel.c4.b3[1].register_forward_hook(hook)
    # myModel.c4.b3[4].register_forward_hook(hook)
    # myModel.c4.b3[7].register_forward_hook(hook)
    # myModel.c4.b4[2].register_forward_hook(hook)
    # myModel.d4.b1[1].register_forward_hook(hook)
    # myModel.d4.b2[1].register_forward_hook(hook)
    # myModel.d4.b2[4].register_forward_hook(hook)
    # myModel.d4.b3[1].register_forward_hook(hook)
    # myModel.d4.b3[4].register_forward_hook(hook)
    # myModel.d4.b3[7].register_forward_hook(hook)
    # myModel.d4.b4[2].register_forward_hook(hook)
    # myModel.e4.b1[1].register_forward_hook(hook)
    # myModel.e4.b2[1].register_forward_hook(hook)
    # myModel.e4.b2[4].register_forward_hook(hook)
    # myModel.e4.b3[1].register_forward_hook(hook)
    # myModel.e4.b3[4].register_forward_hook(hook)
    # myModel.e4.b3[7].register_forward_hook(hook)
    # myModel.e4.b4[2].register_forward_hook(hook)
    # myModel.a5.b1[1].register_forward_hook(hook)
    # myModel.a5.b2[1].register_forward_hook(hook)
    # myModel.a5.b2[4].register_forward_hook(hook)
    # myModel.a5.b3[1].register_forward_hook(hook)
    # myModel.a5.b3[4].register_forward_hook(hook)
    # myModel.a5.b3[7].register_forward_hook(hook)
    # myModel.a5.b4[2].register_forward_hook(hook)
    # myModel.b5.b1[1].register_forward_hook(hook)
    # myModel.b5.b2[1].register_forward_hook(hook)
    # myModel.b5.b2[4].register_forward_hook(hook)
    # myModel.b5.b3[1].register_forward_hook(hook)
    # myModel.b5.b3[4].register_forward_hook(hook)
    # myModel.b5.b3[7].register_forward_hook(hook)
    # myModel.b5.b4[2].register_forward_hook(hook)
    # oriModel.pre_layers[1].register_forward_hook(hook_ori)
    # oriModel.a3.b1[1].register_forward_hook(hook_ori)
    # oriModel.a3.b2[1].register_forward_hook(hook_ori)
    # oriModel.a3.b2[4].register_forward_hook(hook_ori)
    # oriModel.a3.b3[1].register_forward_hook(hook_ori)
    # oriModel.a3.b3[4].register_forward_hook(hook_ori)
    # oriModel.a3.b3[7].register_forward_hook(hook_ori)
    # oriModel.a3.b4[2].register_forward_hook(hook_ori)
    # oriModel.b3.b1[1].register_forward_hook(hook_ori)
    # oriModel.b3.b2[1].register_forward_hook(hook_ori)
    # oriModel.b3.b2[4].register_forward_hook(hook_ori)
    # oriModel.b3.b3[1].register_forward_hook(hook_ori)
    # oriModel.b3.b3[4].register_forward_hook(hook_ori)
    # oriModel.b3.b3[7].register_forward_hook(hook_ori)
    # oriModel.b3.b4[2].register_forward_hook(hook_ori)
    # oriModel.a4.b1[1].register_forward_hook(hook_ori)
    # oriModel.a4.b2[1].register_forward_hook(hook_ori)
    # oriModel.a4.b2[4].register_forward_hook(hook_ori)
    # oriModel.a4.b3[1].register_forward_hook(hook_ori)
    # oriModel.a4.b3[4].register_forward_hook(hook_ori)
    # oriModel.a4.b3[7].register_forward_hook(hook_ori)
    # oriModel.a4.b4[2].register_forward_hook(hook_ori)
    # oriModel.b4.b1[1].register_forward_hook(hook_ori)
    # oriModel.b4.b2[1].register_forward_hook(hook_ori)
    # oriModel.b4.b2[4].register_forward_hook(hook_ori)
    # oriModel.b4.b3[1].register_forward_hook(hook_ori)
    # oriModel.b4.b3[4].register_forward_hook(hook_ori)
    # oriModel.b4.b3[7].register_forward_hook(hook_ori)
    # oriModel.b4.b4[2].register_forward_hook(hook_ori)
    # oriModel.c4.b1[1].register_forward_hook(hook_ori)
    # oriModel.c4.b2[1].register_forward_hook(hook_ori)
    # oriModel.c4.b2[4].register_forward_hook(hook_ori)
    # oriModel.c4.b3[1].register_forward_hook(hook_ori)
    # oriModel.c4.b3[4].register_forward_hook(hook_ori)
    # oriModel.c4.b3[7].register_forward_hook(hook_ori)
    # oriModel.c4.b4[2].register_forward_hook(hook_ori)
    # oriModel.d4.b1[1].register_forward_hook(hook_ori)
    # oriModel.d4.b2[1].register_forward_hook(hook_ori)
    # oriModel.d4.b2[4].register_forward_hook(hook_ori)
    # oriModel.d4.b3[1].register_forward_hook(hook_ori)
    # oriModel.d4.b3[4].register_forward_hook(hook_ori)
    # oriModel.d4.b3[7].register_forward_hook(hook_ori)
    # oriModel.d4.b4[2].register_forward_hook(hook_ori)
    # oriModel.e4.b1[1].register_forward_hook(hook_ori)
    # oriModel.e4.b2[1].register_forward_hook(hook_ori)
    # oriModel.e4.b2[4].register_forward_hook(hook_ori)
    # oriModel.e4.b3[1].register_forward_hook(hook_ori)
    # oriModel.e4.b3[4].register_forward_hook(hook_ori)
    # oriModel.e4.b3[7].register_forward_hook(hook_ori)
    # oriModel.e4.b4[2].register_forward_hook(hook_ori)
    # oriModel.a5.b1[1].register_forward_hook(hook_ori)
    # oriModel.a5.b2[1].register_forward_hook(hook_ori)
    # oriModel.a5.b2[4].register_forward_hook(hook_ori)
    # oriModel.a5.b3[1].register_forward_hook(hook_ori)
    # oriModel.a5.b3[4].register_forward_hook(hook_ori)
    # oriModel.a5.b3[7].register_forward_hook(hook_ori)
    # oriModel.a5.b4[2].register_forward_hook(hook_ori)
    # oriModel.b5.b1[1].register_forward_hook(hook_ori)
    # oriModel.b5.b2[1].register_forward_hook(hook_ori)
    # oriModel.b5.b2[4].register_forward_hook(hook_ori)
    # oriModel.b5.b3[1].register_forward_hook(hook_ori)
    # oriModel.b5.b3[4].register_forward_hook(hook_ori)
    # oriModel.b5.b3[7].register_forward_hook(hook_ori)
    # oriModel.b5.b4[2].register_forward_hook(hook_ori)

    myModel.dense1[0].bn1.register_forward_hook(hook)
    myModel.dense1[0].bn2.register_forward_hook(hook)
    myModel.dense1[1].bn1.register_forward_hook(hook)
    myModel.dense1[1].bn2.register_forward_hook(hook)
    myModel.dense1[2].bn1.register_forward_hook(hook)
    myModel.dense1[2].bn2.register_forward_hook(hook)
    myModel.dense1[3].bn1.register_forward_hook(hook)
    myModel.dense1[3].bn2.register_forward_hook(hook)
    myModel.dense1[4].bn1.register_forward_hook(hook)
    myModel.dense1[4].bn2.register_forward_hook(hook)
    myModel.dense1[5].bn1.register_forward_hook(hook)
    myModel.dense1[5].bn2.register_forward_hook(hook)
    myModel.dense2[0].bn1.register_forward_hook(hook)
    myModel.dense2[0].bn2.register_forward_hook(hook)
    myModel.dense2[1].bn1.register_forward_hook(hook)
    myModel.dense2[1].bn2.register_forward_hook(hook)
    myModel.dense2[2].bn1.register_forward_hook(hook)
    myModel.dense2[2].bn2.register_forward_hook(hook)
    myModel.dense2[3].bn1.register_forward_hook(hook)
    myModel.dense2[3].bn2.register_forward_hook(hook)
    myModel.dense2[4].bn1.register_forward_hook(hook)
    myModel.dense2[4].bn2.register_forward_hook(hook)
    myModel.dense2[5].bn1.register_forward_hook(hook)
    myModel.dense2[5].bn2.register_forward_hook(hook)
    myModel.dense2[6].bn1.register_forward_hook(hook)
    myModel.dense2[6].bn2.register_forward_hook(hook)
    myModel.dense2[7].bn1.register_forward_hook(hook)
    myModel.dense2[7].bn2.register_forward_hook(hook)
    myModel.dense2[8].bn1.register_forward_hook(hook)
    myModel.dense2[8].bn2.register_forward_hook(hook)
    myModel.dense2[9].bn1.register_forward_hook(hook)
    myModel.dense2[9].bn2.register_forward_hook(hook)
    myModel.dense2[10].bn1.register_forward_hook(hook)
    myModel.dense2[10].bn2.register_forward_hook(hook)
    myModel.dense2[11].bn1.register_forward_hook(hook)
    myModel.dense2[11].bn2.register_forward_hook(hook)
    myModel.dense3[0].bn1.register_forward_hook(hook)
    myModel.dense3[0].bn2.register_forward_hook(hook)
    myModel.dense3[1].bn1.register_forward_hook(hook)
    myModel.dense3[1].bn2.register_forward_hook(hook)
    myModel.dense3[2].bn1.register_forward_hook(hook)
    myModel.dense3[2].bn2.register_forward_hook(hook)
    myModel.dense3[3].bn1.register_forward_hook(hook)
    myModel.dense3[3].bn2.register_forward_hook(hook)
    myModel.dense3[4].bn1.register_forward_hook(hook)
    myModel.dense3[4].bn2.register_forward_hook(hook)
    myModel.dense3[5].bn1.register_forward_hook(hook)
    myModel.dense3[5].bn2.register_forward_hook(hook)
    myModel.dense3[6].bn1.register_forward_hook(hook)
    myModel.dense3[6].bn2.register_forward_hook(hook)
    myModel.dense3[7].bn1.register_forward_hook(hook)
    myModel.dense3[7].bn2.register_forward_hook(hook)
    myModel.dense3[8].bn1.register_forward_hook(hook)
    myModel.dense3[8].bn2.register_forward_hook(hook)
    myModel.dense3[9].bn1.register_forward_hook(hook)
    myModel.dense3[9].bn2.register_forward_hook(hook)
    myModel.dense3[10].bn1.register_forward_hook(hook)
    myModel.dense3[10].bn2.register_forward_hook(hook)
    myModel.dense3[11].bn1.register_forward_hook(hook)
    myModel.dense3[11].bn2.register_forward_hook(hook)
    myModel.dense3[12].bn1.register_forward_hook(hook)
    myModel.dense3[12].bn2.register_forward_hook(hook)
    myModel.dense3[13].bn1.register_forward_hook(hook)
    myModel.dense3[13].bn2.register_forward_hook(hook)
    myModel.dense3[14].bn1.register_forward_hook(hook)
    myModel.dense3[14].bn2.register_forward_hook(hook)
    myModel.dense3[15].bn1.register_forward_hook(hook)
    myModel.dense3[15].bn2.register_forward_hook(hook)
    myModel.dense3[16].bn1.register_forward_hook(hook)
    myModel.dense3[16].bn2.register_forward_hook(hook)
    myModel.dense3[17].bn1.register_forward_hook(hook)
    myModel.dense3[17].bn2.register_forward_hook(hook)
    myModel.dense3[18].bn1.register_forward_hook(hook)
    myModel.dense3[18].bn2.register_forward_hook(hook)
    myModel.dense3[19].bn1.register_forward_hook(hook)
    myModel.dense3[19].bn2.register_forward_hook(hook)
    myModel.dense3[20].bn1.register_forward_hook(hook)
    myModel.dense3[20].bn2.register_forward_hook(hook)
    myModel.dense3[21].bn1.register_forward_hook(hook)
    myModel.dense3[21].bn2.register_forward_hook(hook)
    myModel.dense3[22].bn1.register_forward_hook(hook)
    myModel.dense3[22].bn2.register_forward_hook(hook)
    myModel.dense3[23].bn1.register_forward_hook(hook)
    myModel.dense3[23].bn2.register_forward_hook(hook)
    myModel.dense4[0].bn1.register_forward_hook(hook)
    myModel.dense4[0].bn2.register_forward_hook(hook)
    myModel.dense4[1].bn1.register_forward_hook(hook)
    myModel.dense4[1].bn2.register_forward_hook(hook)
    myModel.dense4[2].bn1.register_forward_hook(hook)
    myModel.dense4[2].bn2.register_forward_hook(hook)
    myModel.dense4[3].bn1.register_forward_hook(hook)
    myModel.dense4[3].bn2.register_forward_hook(hook)
    myModel.dense4[4].bn1.register_forward_hook(hook)
    myModel.dense4[4].bn2.register_forward_hook(hook)
    myModel.dense4[5].bn1.register_forward_hook(hook)
    myModel.dense4[5].bn2.register_forward_hook(hook)
    myModel.dense4[6].bn1.register_forward_hook(hook)
    myModel.dense4[6].bn2.register_forward_hook(hook)
    myModel.dense4[7].bn1.register_forward_hook(hook)
    myModel.dense4[7].bn2.register_forward_hook(hook)
    myModel.dense4[8].bn1.register_forward_hook(hook)
    myModel.dense4[8].bn2.register_forward_hook(hook)
    myModel.dense4[9].bn1.register_forward_hook(hook)
    myModel.dense4[9].bn2.register_forward_hook(hook)
    myModel.dense4[10].bn1.register_forward_hook(hook)
    myModel.dense4[10].bn2.register_forward_hook(hook)
    myModel.dense4[11].bn1.register_forward_hook(hook)
    myModel.dense4[11].bn2.register_forward_hook(hook)
    myModel.dense4[12].bn1.register_forward_hook(hook)
    myModel.dense4[12].bn2.register_forward_hook(hook)
    myModel.dense4[13].bn1.register_forward_hook(hook)
    myModel.dense4[13].bn2.register_forward_hook(hook)
    myModel.dense4[14].bn1.register_forward_hook(hook)
    myModel.dense4[14].bn2.register_forward_hook(hook)
    myModel.dense4[15].bn1.register_forward_hook(hook)
    myModel.dense4[15].bn2.register_forward_hook(hook)
    myModel.bn.register_forward_hook(hook)
    oriModel.dense1[0].bn1.register_forward_hook(hook_ori)
    oriModel.dense1[0].bn2.register_forward_hook(hook_ori)
    oriModel.dense1[1].bn1.register_forward_hook(hook_ori)
    oriModel.dense1[1].bn2.register_forward_hook(hook_ori)
    oriModel.dense1[2].bn1.register_forward_hook(hook_ori)
    oriModel.dense1[2].bn2.register_forward_hook(hook_ori)
    oriModel.dense1[3].bn1.register_forward_hook(hook_ori)
    oriModel.dense1[3].bn2.register_forward_hook(hook_ori)
    oriModel.dense1[4].bn1.register_forward_hook(hook_ori)
    oriModel.dense1[4].bn2.register_forward_hook(hook_ori)
    oriModel.dense1[5].bn1.register_forward_hook(hook_ori)
    oriModel.dense1[5].bn2.register_forward_hook(hook_ori)
    oriModel.dense2[0].bn1.register_forward_hook(hook_ori)
    oriModel.dense2[0].bn2.register_forward_hook(hook_ori)
    oriModel.dense2[1].bn1.register_forward_hook(hook_ori)
    oriModel.dense2[1].bn2.register_forward_hook(hook_ori)
    oriModel.dense2[2].bn1.register_forward_hook(hook_ori)
    oriModel.dense2[2].bn2.register_forward_hook(hook_ori)
    oriModel.dense2[3].bn1.register_forward_hook(hook_ori)
    oriModel.dense2[3].bn2.register_forward_hook(hook_ori)
    oriModel.dense2[4].bn1.register_forward_hook(hook_ori)
    oriModel.dense2[4].bn2.register_forward_hook(hook_ori)
    oriModel.dense2[5].bn1.register_forward_hook(hook_ori)
    oriModel.dense2[5].bn2.register_forward_hook(hook_ori)
    oriModel.dense2[6].bn1.register_forward_hook(hook_ori)
    oriModel.dense2[6].bn2.register_forward_hook(hook_ori)
    oriModel.dense2[7].bn1.register_forward_hook(hook_ori)
    oriModel.dense2[7].bn2.register_forward_hook(hook_ori)
    oriModel.dense2[8].bn1.register_forward_hook(hook_ori)
    oriModel.dense2[8].bn2.register_forward_hook(hook_ori)
    oriModel.dense2[9].bn1.register_forward_hook(hook_ori)
    oriModel.dense2[9].bn2.register_forward_hook(hook_ori)
    oriModel.dense2[10].bn1.register_forward_hook(hook_ori)
    oriModel.dense2[10].bn2.register_forward_hook(hook_ori)
    oriModel.dense2[11].bn1.register_forward_hook(hook_ori)
    oriModel.dense2[11].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[0].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[0].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[1].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[1].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[2].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[2].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[3].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[3].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[4].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[4].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[5].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[5].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[6].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[6].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[7].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[7].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[8].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[8].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[9].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[9].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[10].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[10].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[11].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[11].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[12].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[12].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[13].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[13].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[14].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[14].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[15].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[15].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[16].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[16].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[17].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[17].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[18].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[18].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[19].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[19].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[20].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[20].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[21].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[21].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[22].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[22].bn2.register_forward_hook(hook_ori)
    oriModel.dense3[23].bn1.register_forward_hook(hook_ori)
    oriModel.dense3[23].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[0].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[0].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[1].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[1].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[2].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[2].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[3].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[3].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[4].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[4].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[5].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[5].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[6].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[6].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[7].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[7].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[8].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[8].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[9].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[9].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[10].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[10].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[11].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[11].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[12].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[12].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[13].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[13].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[14].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[14].bn2.register_forward_hook(hook_ori)
    oriModel.dense4[15].bn1.register_forward_hook(hook_ori)
    oriModel.dense4[15].bn2.register_forward_hook(hook_ori)
    oriModel.bn.register_forward_hook(hook_ori)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # if batch_idx < 500:
        #     continue
        # if batch_idx >= 1000:
        #     break

        # if batch_idx >= 5000:
        #     break

        if batch_idx < 5000:
            continue
        if batch_idx >= 7500:
            break

        """gradient"""
        x = Variable(inputs, requires_grad=True)
        # optimizer = optim.SGD([x], lr=0.00001, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.SGD([x], lr=0.9)
        # print('start')
        for p in range(10):
            Permutation = torch.zeros(3, 32, 32)
            x_ori = x.data
            v2 = myModel(x.to(device))
            v1 = oriModel(x.to(device))
            # print(len(activation))
            r1 = v1.max(1)[1]
            # print(r1)
            if p == 0:
                ori_target = r1
            else:
                if r1 != ori_target:
                    count_ori_error += 1

            "neuron coverage part"
            while True:
                "vgg"
                # layer = random.randint(0,7)
                "resnet18"
                # layer = random.randint(0,19)
                "googlenet"
                # layer = random.randint(0,63)
                "densenet"
                layer = random.randint(0,116)
                shape_layer = activation[layer].shape
                activation_index = []
                for length in shape_layer:
                    activation_index.append(random.randint(0,length-1))
                activation_value = activation[layer][activation_index[0], activation_index[1], activation_index[2], activation_index[3]]
                activation_value_ori = activation_ori[layer][activation_index[0], activation_index[1], activation_index[2], activation_index[3]]
                # if activation_value <= 0:
                if activation_value <= 0 and activation_value_ori > 0:
                    # print('success')
                    break
            
            # activation_value.backward(retain_graph=True)
            # print(activation_value)
            loss = - (v1[0, r1] - v2[0, r1] + 20 * activation_value)
            # loss = - (v1[0, r1] - v2[0, r1])
            # print(loss)
            # loss = v1[0, r1] - v2[0, r1] + criterion(v1, r1)
            # loss = v1[0, r1] - v2[0, r1]
            # loss = -loss_fn_AE_ce(v1,v2)
            optimizer.zero_grad()
            loss.backward()
            # print(x)
            # for i in range(3):
            #     nn.utils.clip_grad_norm(x[0, i], 0.06, norm_type='inf')
            
            optimizer.step()
            for i in range(3):
                Permutation[i] = torch.clamp(x[0, i] - x_ori[0, i], min=-var[i], max=var[i])
            x.data = x_ori + Permutation
            """min max"""
            x_copy = deepcopy(x[0].cpu().detach().float().numpy())
            # for i in range(3):
            #     x[0, i] = torch.clamp(x[0, i], min= (0 - mean[i])/ std[i], max= (1 - mean[i])/ std[i])
            # x = Variable(x, requires_grad=True)

            save_tensor(x_copy, batch_idx, p, root_path2, ori_target.item())

            # grad_orig = x.grad.data.cpu().numpy().copy()
            # print(grad_orig.max())
            # print(grad_orig.min())
            activation.clear()
            activation_ori.clear()
            gc.collect()
            torch.cuda.empty_cache()

        """ used for testing error made by deepxplore in original model"""
        v1 = oriModel(x.to(device))
        r1 = v1.max(1)[1]
        if r1 != ori_target:
            count_ori_error += 1
        activation_ori.clear()
        # break
    print(count_ori_error)
    



if __name__ == "__main__":
    main()