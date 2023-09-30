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

import cv2
import shutil 
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

def load_model(model_file):
    # model = VGG_no_sequntial_relu('VGG11')
    # model = VGG('VGG11')
    # model = ResNet18()
    # model = GoogLeNet_relu()
    # model = DenseNet121()
    model = VGG('VGG16')
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['net'])
    model.to(device)
    return model

# def load_model_2(model_file):
#     # model = VGG_no_sequntial_relu('VGG11')
#     model = ResNet18()
#     state_dict = torch.load(model_file)
#     new_state_dict = OrderedDict()
#     for k, v in state_dict['state_dict'].items():
#         name = k[7:]
#         new_state_dict[name] = v
#     model.load_state_dict(new_state_dict)
#     model.to(device)
#     return model

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

def save_tensor(inputs, batch_idx, root_path):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    image_numpy = inputs[0].cpu().detach().float().numpy()

    for i in range(len(mean)):
        image_numpy[i] = image_numpy[i] * std[i] + mean[i]
    
    image_numpy = image_numpy * 255
    image_numpy = np.transpose(image_numpy, (1, 2, 0)).astype(np.uint8)
    cv2.imwrite(root_path + str(batch_idx) + '.jpg', image_numpy)

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
            a = cv2.imread(path)
            self.frames.append((transform_test(cv2.imread(path)), int(path[-5]), path))

    
    def __getitem__(self, index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)

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
    testset_test = torchvision.datasets.CIFAR10(
        root='../pytorch-cifar/data', train=False, download=True, transform=transform_test)
    testloader_test = torch.utils.data.DataLoader(
        testset_test, batch_size=100, shuffle=False, num_workers=2)
    testloader_ori = torch.utils.data.DataLoader(
        testset_test, batch_size=1, shuffle=False, num_workers=2)
    testiter = iter(testloader_ori)
    # print(type(next(testiter)[0]))
    # testset = Dataset_Diff('/home/ylipf/Model-Compression-testing/2PruningDiffDeepGra_vgg_rate75/')
    # testset = Dataset_Diff('2PruningDiffDeepGra_vgg_rate75_20nc/')
    # testset = Dataset_Diff('2PruningDiffDeepGra_resnet_rate75/')
    # testset = Dataset_Diff('2PruningDiffDeepGra_resnet_rate75_10nc/')
    # testset = Dataset_Diff('2PruningDiffDeepGra_google_rate75/')
    # testset = Dataset_Diff('2PruningDiffDeepGra_densenet_rate75/')
    testset = Dataset_Diff('2PruningDiffDeepGra_vgg16_rate75/')

    # testset = Dataset_Diff('2PruningDiffDeepGra_densenet_rate75_test/')
    
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    # myModel = load_model('../pytorch-cifar/checkpoint_relu_1113/ckpt.pth').to(device)
    # myModel = load_model('../pytorch-cifar/checkpoint_resnet18/ckpt.pth').to(device)
    # myModel = load_model('../pytorch-cifar/checkpoint_googlenet_normal/ckpt.pth').to(device)
    myModel = load_model('../pytorch-cifar/checkpoint_vgg16/ckpt.pth').to(device)
    print('success')
    # print(myModel.state_dict().keys())
    # print(myModel)
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
    #     (myModel.a3.b2[0], 'weight'),
    #     (myModel.a3.b2[3], 'weight'),
    #     (myModel.a3.b3[0], 'weight'),
    #     (myModel.a3.b3[3], 'weight'),
    #     (myModel.a3.b3[6], 'weight'),
    #     (myModel.a3.b4[1], 'weight'),
    #     (myModel.b3.b1[0], 'weight'),
    #     (myModel.b3.b2[0], 'weight'),
    #     (myModel.b3.b2[3], 'weight'),
    #     (myModel.b3.b2[0], 'weight'),
    #     (myModel.b3.b2[3], 'weight'),
    #     (myModel.b3.b3[0], 'weight'),
    #     (myModel.b3.b3[3], 'weight'),
    #     (myModel.b3.b3[6], 'weight'),
    #     (myModel.b3.b4[1], 'weight'),
    #     (myModel.a4.b1[0], 'weight'),
    #     (myModel.a4.b2[0], 'weight'),
    #     (myModel.a4.b2[3], 'weight'),
    #     (myModel.a4.b2[0], 'weight'),
    #     (myModel.a4.b2[3], 'weight'),
    #     (myModel.a4.b3[0], 'weight'),
    #     (myModel.a4.b3[3], 'weight'),
    #     (myModel.a4.b3[6], 'weight'),
    #     (myModel.a4.b4[1], 'weight'),
    #     (myModel.b4.b1[0], 'weight'),
    #     (myModel.b4.b2[0], 'weight'),
    #     (myModel.b4.b2[3], 'weight'),
    #     (myModel.b4.b2[0], 'weight'),
    #     (myModel.b4.b2[3], 'weight'),
    #     (myModel.b4.b3[0], 'weight'),
    #     (myModel.b4.b3[3], 'weight'),
    #     (myModel.b4.b3[6], 'weight'),
    #     (myModel.b4.b4[1], 'weight'),
    #     (myModel.c4.b1[0], 'weight'),
    #     (myModel.c4.b2[0], 'weight'),
    #     (myModel.c4.b2[3], 'weight'),
    #     (myModel.c4.b2[0], 'weight'),
    #     (myModel.c4.b2[3], 'weight'),
    #     (myModel.c4.b3[0], 'weight'),
    #     (myModel.c4.b3[3], 'weight'),
    #     (myModel.c4.b3[6], 'weight'),
    #     (myModel.c4.b4[1], 'weight'),
    #     (myModel.d4.b1[0], 'weight'),
    #     (myModel.d4.b2[0], 'weight'),
    #     (myModel.d4.b2[3], 'weight'),
    #     (myModel.d4.b2[0], 'weight'),
    #     (myModel.d4.b2[3], 'weight'),
    #     (myModel.d4.b3[0], 'weight'),
    #     (myModel.d4.b3[3], 'weight'),
    #     (myModel.d4.b3[6], 'weight'),
    #     (myModel.d4.b4[1], 'weight'),
    #     (myModel.e4.b1[0], 'weight'),
    #     (myModel.e4.b2[0], 'weight'),
    #     (myModel.e4.b2[3], 'weight'),
    #     (myModel.e4.b2[0], 'weight'),
    #     (myModel.e4.b2[3], 'weight'),
    #     (myModel.e4.b3[0], 'weight'),
    #     (myModel.e4.b3[3], 'weight'),
    #     (myModel.e4.b3[6], 'weight'),
    #     (myModel.e4.b4[1], 'weight'),
    #     (myModel.a5.b1[0], 'weight'),
    #     (myModel.a5.b2[0], 'weight'),
    #     (myModel.a5.b2[3], 'weight'),
    #     (myModel.a5.b2[0], 'weight'),
    #     (myModel.a5.b2[3], 'weight'),
    #     (myModel.a5.b3[0], 'weight'),
    #     (myModel.a5.b3[3], 'weight'),
    #     (myModel.a5.b3[6], 'weight'),
    #     (myModel.a5.b4[1], 'weight'),
    #     (myModel.b5.b1[0], 'weight'),
    #     (myModel.b5.b2[0], 'weight'),
    #     (myModel.b5.b2[3], 'weight'),
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
    # state_dict = torch.load('2checkpoint_vgg_finetune_diff_train_rate75_20nc/ckpt.pth')
    # state_dict = torch.load('2checkpoint_resnet_finetune_rate75/ckpt.pth')
    # state_dict = torch.load('2checkpoint_resnet_finetune_diff_train_rate75/ckpt.pth')
    # state_dict = torch.load('2checkpoint_google_finetune_rate75/ckpt.pth')
    # state_dict = torch.load('2checkpoint_google_finetune_diff_train_rate75_1/ckpt.pth')
    # state_dict = torch.load('2checkpoint_vgg16_finetune_rate75/ckpt.pth')
    state_dict = torch.load('2checkpoint_vgg162_finetune_diff_train_rate75/ckpt_final.pth')
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
    # for parameters in oriModel.parameters():
    #     print(parameters)

    # root_path1 = '/home/ylipf/Model-Compression-testing/PruningDiff_no_mutate/'
    # os.makedirs(root_path1, exist_ok = True)
    # root_path2 = '/home/ylipf/Model-Compression-testing/PruningDiff/'
    # os.makedirs(root_path2, exist_ok = True)
    different_input = 0
    idx = 0
    change_ori = 0
    l2 = 0
    for idx, (inputs, ori_target, path) in enumerate(testloader):
        # if idx == 0:
        #     seg = path[0].split('/')
        #     seg[-2] += '_maxnoAE'
        #     root_path = '/'.join(seg[:-1])
        #     os.makedirs(root_path, exist_ok = True)
        if idx % 10 == 0:
            beforemut = next(testiter)[0]
        v2 = myModel(inputs.to(device))
        ori_target = ori_target.to(device)
        r2 = v2.max(1)[1]
        v1 = oriModel(inputs.to(device))
        r1 = v1.max(1)[1]
        if r2 != r1:
            different_input += 1
            if r1 != ori_target:
                change_ori += 1
            else:
                l2 += torch.norm(inputs-beforemut, 2) / (3 * 32 * 32)
            # else:
                # seg = path[0].split('/')
                # seg[-2] += '_maxnoAE'
                # new_path = '/'.join(seg)
                # shutil.copyfile(path[0], new_path)    
            
    print(different_input)
    print(idx)
    print(different_input/(idx + 1))
    print(change_ori)
    print('l2:', l2/(different_input - change_ori))




if __name__ == "__main__":
    main()