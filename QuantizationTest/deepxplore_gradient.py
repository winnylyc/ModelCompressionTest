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
from torch.autograd import Variable
import torch.optim as optim
from copy import deepcopy
import gc

import cv2 
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
    torch.manual_seed(2)
    torch.cuda.manual_seed(2)
    np.random.seed(2)
    random.seed(2)  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True, transform=transform_test)
    testloader_test = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

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

    oriModel = resnet18_relu()
    oriModel.load_state_dict(torch.load(model_filepath, map_location=device)['net'])
    oriModel.to(device)
                                                 
    oriModel.eval()
    # acc_p = test(0, myModel, testloader_test)
    # acc_o = test(0, oriModel, testloader_test)
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
    root_path2 = 'QuanDiffDeepGra_resnet18_normal_test/'
    os.makedirs(root_path2, exist_ok = True)
    var = [0.0608027,  0.05892733, 0.06850188]
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    loss_fn_AE_ce = nn.CrossEntropyLoss(reduction = 'none')
    count_ori_error = 0
    criterion = nn.CrossEntropyLoss()
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
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # if batch_idx < 500:
        #     continue
        # if batch_idx >= 1000:
        #     break

        if batch_idx >= 5000:
            break

        # if batch_idx < 5000:
        #     continue
        # if batch_idx >= 7500:
        #     break

        # v2 = myModel(inputs.to(device))
        # layer2 = activation[0].cpu().detach().numpy()
        # v1 = oriModel(inputs.to(device))
        # layer1 = activation_ori[0].cpu().detach().numpy()

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
            r1 = v1.max(1)[1]
            # print(r1)
            if p == 0:
                ori_target = r1
            else:
                if r1 != ori_target:
                    count_ori_error += 1
            # print(v1[0, r1])
            # print(v2[0, r1])
            # print(len(activation_ori))
            # print(len(activation))
            # layer = random.randint(0,7)
            # shape_layer = activation[layer].shape
            # print(shape_layer)

            "neuron coverage part"
            # while True:
            #     layer = random.randint(0,7)
            #     shape_layer = activation[layer].shape
            #     activation_index = []
            #     for length in shape_layer:
            #         activation_index.append(random.randint(0,length-1))
            #     activation_value = activation[layer][activation_index[0], activation_index[1], activation_index[2], activation_index[3]]
            #     activation_value_ori = activation_ori[layer][activation_index[0], activation_index[1], activation_index[2], activation_index[3]]
            #     # if activation_value <= 0:
            #     if activation_value <= 0 and activation_value_ori > 0:
            #         # print('success')
            #         break
            
            # activation_value.backward(retain_graph=True)
            # print(activation_value)
            # loss = - (v1[0, r1] - v2[0, r1] + activation_value)
            loss = - (v1[0, r1] - v2[0, r1])
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
            # for i in range(3):
            #     x.data[0, i] = torch.clamp(x[0, i], min= 0, max= 1)
            x_copy = deepcopy(x[0].cpu().detach().float().numpy())

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