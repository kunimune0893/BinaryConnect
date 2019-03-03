#!/usr/bin/env python
# coding: utf-8

# # 参考URL
# - Residual Network(ResNet)の理解とチューニングのベストプラクティス：[https://deepage.net/deep_learning/2016/11/30/resnet.html](https://deepage.net/deep_learning/2016/11/30/resnet.html)

# # ソース
# ## import
import os, argparse, math
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

print( torch.__version__ )

# ## parameter
params = {}
params['epoch']  = 10
params['start_epoch'] = 0
params['batch']  = 32
params['lr']     = 0.1
#params['resume'] = False
params['resume'] = "./runs/WideResNet-28-10/model_best.pth.tar"
params['tbdir']  = "./"

# ## class BasicBlock()
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

# ## class NetworkBlock()
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

# ## class WideResNet()
class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

# ## class BC()
class BC():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 0
        end_range = count_targets-1
        self.bin_range = np.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        self.save_params()
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.target_modules[index].data.sign())

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)


    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):
        clip_scale=[]
        m=nn.Hardtanh(-1, 1)
        for index in range(self.num_of_params):
            clip_scale.append(m(Variable(self.target_modules[index].data)))
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(clip_scale[index].data)

# ## main()

params['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params['model'] = WideResNet( 28, num_classes=10, widen_factor=10, dropRate=0.0 )

print( 'Number of model parameters: {}'.format(sum([p.data.nelement() for p in params['model'].parameters()])) )

params['model'] = params['model'].to( params['device'] )
if params['device'] == "cuda:0":
    params['model'] = torch.nn.DataParallel( params['model'] )
    cudnn.benchmark = True

bin_op = BC( params['model'] )

criterion = nn.CrossEntropyLoss()

if params['resume'] and os.path.isfile( params['resume'] ):
    print( "=> loading checkpoint '{}'".format(params['resume']) )
    checkpoint = torch.load( params['resume'] )
    params['start_epoch'] = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    params['model'].load_state_dict( checkpoint['state_dict'] )
    print( "=> loaded checkpoint '{}' (epoch {})".format(params['resume'], checkpoint['epoch']) )
else:
    print( "=> no checkpoint found at '{}'".format(params['resume']) )

transform_tst = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
tst_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transform_tst),
    batch_size=params['batch'], shuffle=False, num_workers=1, pin_memory=True )

params['model'].eval()
bin_op.binarization()

for ii, (name, mm) in enumerate(params['model'].named_modules()):
    if "block1.layer.0.conv1" in (name):
        print( "ii=", ii, ": name=", name, ", type=", type(mm), mm )
        print( "mm.weight.shape=", mm.weight.shape )
        print( "mm.weight.shape=", mm.weight[0])

loss = 0
total = 0
correct = 0
with torch.no_grad():
    for ii, (input, label) in enumerate(tst_loader):
        input, label = input.to(params['device']), label.to(params['device'])
        output = params['model'](input)
        loss += criterion(output, label).item()
        _, predict = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predict == label).sum().item()

print( "tst loss={:.3f}, acc={:.3f}% ({:5d}/{:5d})".format(
        loss, correct / total * 100, correct, total) )

