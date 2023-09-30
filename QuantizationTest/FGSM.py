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

from resnet_relu import resnet18_relu
import shutil 

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def test(epoch, net, testloader):
    global best_acc
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
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

def fgsm_attack(image, epsilon, data_grad):
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    sign_data_grad = data_grad.sign()
    # 通过epsilon生成对抗样本
    # print(']]]]]]]]]]]]]]]]]]]]]]]]]]]]')
    # print(image.min())
    # print(image.max())
    perturbed_image = torch.empty((1, 3, 32, 32)).to(device)
    perturbed_image[0, 0] = image[0, 0] + epsilon*sign_data_grad[0, 0]*(1/255/0.2023)
    perturbed_image[0, 1] = image[0, 1] + epsilon*sign_data_grad[0, 1]*(1/255/0.1994)
    perturbed_image[0, 2] = image[0, 2] + epsilon*sign_data_grad[0, 2]*(1/255/0.2010)
    # print(perturbed_image.min())
    # print(perturbed_image.max())
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    # print(perturbed_image.shape)
    perturbed_image[0, 0] = torch.clamp(perturbed_image[0, 0], (-0.4914/0.2023), ((1-0.4914)/0.2023))
    perturbed_image[0, 1] = torch.clamp(perturbed_image[0, 1], (-0.4822/0.1994), ((1-0.4822)/0.1994))
    perturbed_image[0, 2] = torch.clamp(perturbed_image[0, 2], (-0.4465/0.2010), ((1-0.4465)/0.2010))
    # 返回对抗样本
    return perturbed_image

def test_fgsm( model, model_pruned, device, test_loader, epsilon ):
 
    # 准确度计数器
    correct = 0
    # 对抗样本
    adv_examples = []

    criterion = nn.CrossEntropyLoss()
    # 循环所有测试集
    for batch_idx, (data, target) in enumerate(test_loader):
        # 将数据和标签发送到设备
        if batch_idx < 5000:
            continue
        data, target = data.to(device), target.to(device)
 
        # 设置张量的requires_grad属性。重要的攻击
        data_test = data.detach()
    
        output_pruned = model_pruned(data_test)
        init_pred = output_pruned.max(1, keepdim=True)[1] # 得到最大对数概率的索引
        
 
        # 如果最初的预测是错误的，不要再攻击了，继续下一个目标的对抗训练
        if init_pred.item() != target.item():
            continue
            
        data.requires_grad = True

        # 通过模型向前传递数据
        output = model(data)
 
        # 计算损失
        loss = criterion(output, target)
 
        # 使所有现有的梯度归零
        model.zero_grad()
 
        # 计算模型的后向梯度
        loss.backward()
 
        # 收集datagrad
        data_grad = data.grad.data
 
        # 调用FGSM攻击
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
 
        # 对受扰动的图像进行重新分类
        output = model_pruned(perturbed_data)
 
        # 检查是否成功
        final_pred = output.max(1, keepdim=True)[1] # 得到最大对数概率的索引
        if final_pred.item() == target.item():
            correct += 1
        #     # 这里都是为后面的可视化做准备
        #     if (epsilon == 0) and (len(adv_examples) < 5):
        #         adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        #         adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        # else:
        #     # 这里都是为后面的可视化做准备
        #     if len(adv_examples) < 5:
        #         adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        #         adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
 
    # 计算最终精度
    final_acc = correct/5000
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, 5000, final_acc))
 
    # 返回准确性和对抗性示例
    return final_acc

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
    testset_test = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=transform_test)
    testloader_test = torch.utils.data.DataLoader(
        testset_test, batch_size=1, shuffle=False, num_workers=2)
    
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
    myModel.load_state_dict(torch.load('checkpoint_resnet18Q_finetune_diff_train_normal/ckpt.pth', map_location=device)['net'])
    # myModel.load_state_dict(torch.load(quantized_model_filepath, map_location=device))

    myModel.eval()

    prunedModel = torch.quantization.convert(myModel, inplace=False)
    # prunedModel = myModel

    prunedModel.eval()

    # oriModel = resnet18_relu()
    # oriModel.load_state_dict(torch.load(model_filepath, map_location=device)['net'])
    # oriModel.to(device)
                                                 
    # oriModel.eval()
    # acc_p = test(0, myModel, testloader_test)
    # acc_o = test(0, oriModel, testloader_test)
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
    # different_input = 0
    # idx = 0
    # change_ori = 0

    accuracies = []
    examples = []
    
    # 对每个干扰程度进行测试
    
    epsilons = [2, 8, 16]
    for eps in epsilons:
        acc = test_fgsm(myModel, prunedModel, device, testloader_test, eps)
        accuracies.append(acc*100)

if __name__ == "__main__":
    main()