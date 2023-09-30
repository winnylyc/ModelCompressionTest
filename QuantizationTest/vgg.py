'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

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

class VGG_no_sequntial_quan_ori(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_no_sequntial_quan_ori, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
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
            nn.ReLU(inplace=False)
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
            nn.ReLU(inplace=False)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)
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
            nn.ReLU(inplace=False)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False)
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
            nn.ReLU(inplace=False)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False)
        )
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(kernel_size=1, stride=1)       
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.quant(x)
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
        out = self.dequant(out)
        return out

class VGG_no_sequntial_quan(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_no_sequntial_quan, self).__init__()
        self.quant1 = torch.quantization.QuantStub()
        self.dequant1 = torch.quantization.DeQuantStub()
        self.quant2 = torch.quantization.QuantStub()
        self.dequant2 = torch.quantization.DeQuantStub()
        self.quant3 = torch.quantization.QuantStub()
        self.dequant3 = torch.quantization.DeQuantStub()
        self.quant4 = torch.quantization.QuantStub()
        self.dequant4 = torch.quantization.DeQuantStub()
        self.quant5 = torch.quantization.QuantStub()
        self.dequant5 = torch.quantization.DeQuantStub()
        self.quant6 = torch.quantization.QuantStub()
        self.dequant6 = torch.quantization.DeQuantStub()
        self.quant7 = torch.quantization.QuantStub()
        self.dequant7 = torch.quantization.DeQuantStub()
        self.quant8 = torch.quantization.QuantStub()
        self.dequant8 = torch.quantization.DeQuantStub()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
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
            nn.ReLU(inplace=False)
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
            nn.ReLU(inplace=False)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)
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
            nn.ReLU(inplace=False)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False)
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
            nn.ReLU(inplace=False)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False)
        )
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(kernel_size=1, stride=1)       
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.quant1(x)
        x = self.conv1(x)
        x = self.dequant1(x)
        x = self.max1(x)
        x = self.quant2(x)
        x = self.conv2(x)
        x = self.dequant2(x)
        x = self.max2(x)
        x = self.quant3(x)
        x = self.conv3(x)
        x = self.dequant2(x)
        x = self.quant4(x)
        x = self.conv4(x)
        x = self.dequant4(x)
        x = self.max3(x)
        x = self.quant5(x)
        x = self.conv5(x)
        x = self.dequant5(x)
        x = self.quant6(x)
        x = self.conv6(x)
        x = self.dequant6(x)
        x = self.max4(x)
        x = self.quant7(x)
        x = self.conv7(x)
        x = self.dequant7(x)
        x = self.quant8(x)
        x = self.conv8(x)
        x = self.dequant8(x)
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

class VGGQ(nn.Module):
    def __init__(self, vgg_name):
        super(VGGQ, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.dequant(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.ReLU(inplace=False)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGGQ2(nn.Module):
    def __init__(self, vgg_name):
        super(VGGQ2, self).__init__()
        self.quant1 = torch.quantization.QuantStub()
        self.features = self._make_layers(cfg[vgg_name])
        self.quant2 = torch.quantization.QuantStub()
        self.classifier = nn.Linear(512, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant1(x)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        x = self.quant2(x)
        out = self.classifier(out)
        out = self.dequant(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += torch.quantization.DeQuantStub()
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if i != len(cfg) - 1:
                    layers += torch.quantization.QuantStub()
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=False)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
