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
import shutil 
from collections import OrderedDict

from vgg import VGG

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def load_model(model_file):
    # model = VGG_no_sequntial_relu('VGG11')
    model = VGG('VGG16')
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
            self.frames.append((transform_test(cv2.imread(path)), int(path[-5]), path))

    
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

def main():
    print('==> Preparing data..')
    # torch.manual_seed(2)
    # torch.cuda.manual_seed(2)
    # np.random.seed(2)
    # random.seed(2)  

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset_test = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)
    testloader_test = torch.utils.data.DataLoader(
        testset_test, batch_size=100, shuffle=False, num_workers=2)
    testloader_ori = torch.utils.data.DataLoader(
        testset_test, batch_size=1, shuffle=False, num_workers=2)
    testiter = iter(testloader_ori)
    testset = Dataset_Diff('QuanDiffDeepGra_vgg16_test/')
    
    
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    model_dir = "saved_models_vgg16"
    pre_quantized_model_filename = "vgg16_pre_convert_cifar10.pt"
    model_filepath = 'checkpoint_vgg16/ckpt.pth'
    quantized_model_filepath = os.path.join(model_dir,
                                            pre_quantized_model_filename)
    fused_model = VGG('VGG16')
    fused_model.load_state_dict(torch.load(model_filepath, map_location=device)['net'])
    fused_model.to(device)
    fused_model.train()
    fused_model = torch.quantization.fuse_modules(fused_model,
                                                  [["features.0", "features.1", "features.2"], ["features.3", "features.4", "features.5"],
                                                  ["features.7", "features.8", "features.9"], ["features.10", "features.11", "features.12"],
                                                  ["features.14", "features.15", "features.16"], ["features.17", "features.18", "features.19"],
                                                  ["features.20", "features.21", "features.22"], ["features.27", "features.28", "features.29"],
                                                  ["features.30", "features.31", "features.32"],["features.34", "features.35", "features.36"],
                                                  ["features.37", "features.38", "features.39"],["features.40", "features.41", "features.42"],],
                                                  inplace=True)
    fused_model.eval()
    myModel = QuantizedResNet18(model_fp32=fused_model)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    myModel.qconfig = quantization_config
    torch.quantization.prepare_qat(myModel, inplace=True)
    # myModel.load_state_dict(torch.load('checkpoint_densenetQ_finetune_diff_train_normal/ckpt.pth', map_location=device)['net'])
    myModel.load_state_dict(torch.load(quantized_model_filepath, map_location=device))

    myModel = torch.quantization.convert(myModel, inplace=True)

    # quantized_model.eval()

    myModel.eval()

    oriModel = VGG('VGG16')
    oriModel.load_state_dict(torch.load(model_filepath, map_location=device)['net'])
    oriModel.to(device)
                                                 
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
        # print(idx)
        # if idx == 5000:
        #     break
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