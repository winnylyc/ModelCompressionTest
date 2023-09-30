# import matplotlib.pyplot as plt

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
import torch.optim as optim

import torch.nn.functional as F

from collections import OrderedDict

from resnet_relu import resnet18_relu

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def load_model(model_file):
    # model = VGG_no_sequntial_relu('VGG11')
    model = ResNet18()
    state_dict = torch.load(model_file)
    if 'state_dict' in state_dict.keys():
        state = 'state_dict'
        model.load_state_dict(state_dict[state])
    elif 'net' in state_dict.keys():
        state = 'net'
        model.load_state_dict(state_dict[state])
    else:
        model.load_state_dict(state_dict)
    # model.load_state_dict(state_dict[state])
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

def save_tensor(inputs, batch_idx, root_path):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    image_numpy = inputs[0].cpu().detach().float().numpy()

    for i in range(len(mean)):
        image_numpy[i] = image_numpy[i] * std[i] + mean[i]
    
    image_numpy = image_numpy * 255
    image_numpy = np.transpose(image_numpy, (1, 2, 0)).astype(np.uint8)
    cv2.imwrite(root_path + str(batch_idx) + '.jpg', image_numpy)

class Dataset_finetune_Diff(data.Dataset):
    def __init__(self, path):
        self.frames = []
        self.setup(path)
    
    def setup(self, path):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        model_dir = "saved_models_resnet18_normal"
        model_filename = "resnet18_cifar10.pt"
        quantized_model_filename = "resnet18_pre_convert_cifar10.pt"
        model_filepath = 'checkpoint_resnet18relu_normal/ckpt.pth'
        oriModel = resnet18_relu()
        oriModel.load_state_dict(torch.load(model_filepath, map_location=device)['net'])
        oriModel.to(device)
                                                        
        oriModel.eval()
        for paths in glob(path + '/*.jpg'):
            # print(paths)
            x = transform_test(cv2.imread(paths))
            # print(x.shape)
            # print(type(x))
            target_ori = int(paths[-5])
            outputs = oriModel(torch.unsqueeze(x, 0).to(device))
            target = outputs.max(1)[1].item()
            if target == target_ori:
                self.frames.append((x, target_ori))
        

    
    def __getitem__(self, index):
        return self.frames[index]

    def __len__(self):
        return len(self.frames)

class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):

        super(QuantizedResNet18, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def main():
    print('==> Preparing data..')
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    np.random.seed(2)
    random.seed(2)  
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # trainset = torchvision.datasets.CIFAR10(
    #     root='/home/ylipf/pytorch-cifar/data', train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)
    testloader_test = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)

    # trainset = Dataset_Diff('/home/ylipf/Integration/PruningDiffDeepGra_resnet18AT_rate75_maxnoAE/')
    trainset = Dataset_finetune_Diff('QuanDiffDeepGra_resnet18_normal/')
    # trainloader = torch.utils.data.DataLoader(
        # trainset, batch_size=50, shuffle=False, num_workers=2)
    trainset_train = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True, transform=transform_train)
    # trainloader_train = torch.utils.data.DataLoader(
    #     trainset_train, batch_size=100, shuffle=False, num_workers=2)
    # trainloader = torch.utils.data.DataLoader(
    #          ConcatDataset(
    #              trainset,
    #              trainset_train
    #          ),
    #          batch_size=128, shuffle=True,
    #          num_workers=2)
    trainset_cat = torch.utils.data.ConcatDataset([trainset,trainset_train])
    trainloader = torch.utils.data.DataLoader(
        trainset_cat, batch_size=128, shuffle=True, num_workers=2)
    

    model_dir = "saved_models_resnet18_normal"
    model_filename = "resnet18_cifar10.pt"
    quantized_model_filename = "resnet18_pre_convert_cifar10.pt"
    model_filepath = 'checkpoint_resnet18relu_normal/ckpt.pth'
    quantized_model_filepath = os.path.join(model_dir,
                                            quantized_model_filename)
    fused_model = resnet18_relu()
    fused_model.load_state_dict(torch.load(model_filepath, map_location=device)['net'])
    fused_model.to(device)
    fused_model.train()
    fused_model = torch.quantization.fuse_modules(fused_model,
                                                  [["conv1", "bn1", "relu"]],
                                                  inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(
                    basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                    inplace=True)
                # for sub_block_name, sub_block in basic_block.named_children():
                #     if sub_block_name == "downsample":
                #         torch.quantization.fuse_modules(sub_block,
                #                                         [["0", "1"]],
                #                                         inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "shortcut" and len(sub_block) > 0:
                        torch.quantization.fuse_modules(sub_block,[["0", "1"]], inplace=True)
    fused_model.eval()
    myModel = QuantizedResNet18(model_fp32=fused_model)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    myModel.qconfig = quantization_config
    torch.quantization.prepare_qat(myModel, inplace=True)
    myModel.load_state_dict(torch.load(quantized_model_filepath, map_location=device))

    myModel.eval()
    print('success')
    acc_p = test(0, myModel, testloader_test)
    criterion_test = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.KLDivLoss()
    """rate 0.85"""
    # optimizer = optim.SGD(myModel.parameters(), lr=0.001,
    #                     momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(myModel.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    best_acc = 0
    for epoch in range(50):
        myModel.train()
        correct = 0
        total = 0
        loss_count = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = myModel(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            # print(loss)
            optimizer.step()
            _, predicted = outputs.max(1)
            predicted_tar = targets
            # _, predicted_tar = targets.max(1)
            total += targets.size(0)
            correct += predicted.eq(predicted_tar).sum().item()
            loss_count += loss.sum().item()
        # for batch_idx, (inputs, targets) in enumerate(trainloader_train):
        #     inputs, targets = inputs.to(device), targets.to(device)
        #     optimizer.zero_grad()
        #     outputs = myModel(inputs)
        #     loss = criterion(outputs, targets)
        #     loss.backward()
        #     optimizer.step()
        #     _, predicted = outputs.max(1)
        #     total += targets.size(0)
        #     correct += predicted.eq(targets).sum().item()
        print(total)
        print(epoch, ':')
        print('Train set accuracy:', 100.*correct/total)
        print('Train set loss:', loss_count/total)
        # scheduler.step()
        myModel.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader_test):
                if batch_idx < 50:
                    continue
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = myModel(inputs)
                loss = criterion_test(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        print('Test set accuracy:', acc)
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': myModel.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint_resnet18Q_finetune_diff_train_normal'):
                os.mkdir('checkpoint_resnet18Q_finetune_diff_train_normal')
            torch.save(state, './checkpoint_resnet18Q_finetune_diff_train_normal/ckpt.pth')
            best_acc = acc
        print('Test set accuracy:', acc)
    torch.save(state, './checkpoint_resnet18Q_finetune_diff_train_normal/ckpt_final.pth')
        
if __name__ == "__main__":
    main()