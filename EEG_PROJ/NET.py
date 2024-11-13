# 2023/07/28 zjn: 把输入数据改为16*63
# 2023/07/29 zyt: add linear network

import torch
from   torch import nn
from   torch.nn import functional as F


# CSAC NET  ***************     CSAC NET  ***************       CSAC NET  ***************
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class CSAC_cell(nn.Module):
    """
    CSAC block :input size:[benchsize, 128, 33, 33](STFT)
    """
    def __init__(self, ch_in, ch_out, k1, k2):
        """
        :param ch_in:
        :param ch_out:
        :param stride:
        """
        super(CSAC_cell, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=k1)
        self.CBAM = CBAMLayer(ch_out)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=k2, padding=1)  # padding = k2 is given, but maybe something wrong?

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """

        x = self.conv(x)
        x = self.CBAM(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class CSAC_NET(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(CSAC_NET, self).__init__()

        # followed 3 blocks
        # [b, 128, h, w] => [b, 256, h/2 ,w/2]
        self.blk1 = CSAC_cell(ch_in, 256, 3, 2)
        # [b, 256, h/2, w/2] => [b, 512, h/4, w/4]
        self.blk2 = CSAC_cell(256, 512, 3, 2)
        # # [b, 512, h/4, w/4] => [b, 512, h/4, w/4]
        self.blk3 = CSAC_cell(512, ch_out, 3, 2)
        # # [b, 512, h/4, w/4] => [b,-1]
        self.outlayer = nn.Linear(ch_out*2*7, 4)  # parameters will be changed

    def forward(self, x):

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x

class CSAC_NET_fff(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(CSAC_NET_fff, self).__init__()

        # followed 3 blocks
        # [b, 128, h, w] => [b, 256, h/2 ,w/2]
        self.blk1 = CSAC_cell(ch_in, 256, 3, 2)
        # [b, 256, h/2, w/2] => [b, 512, h/4, w/4]
        self.blk2 = CSAC_cell(256, 512, 3, 2)
        # # [b, 512, h/4, w/4] => [b, 512, h/4, w/4]
        self.blk3 = CSAC_cell(512, ch_out, 3, 2)
        # # [b, 512, h/4, w/4] => [b,-1]
        self.outlayer = nn.Linear(ch_out*2*7, 7)  # parameters will be changed

    def forward(self, x):

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x

class CSAC_NET_udlsrn(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(CSAC_NET_udlsrn, self).__init__()

        # followed 3 blocks
        # [b, 128, h, w] => [b, 256, h/2 ,w/2]
        self.blk1 = CSAC_cell(ch_in, 256, 3, 2)
        # [b, 256, h/2, w/2] => [b, 512, h/4, w/4]
        self.blk2 = CSAC_cell(256, 512, 3, 2)
        # # [b, 512, h/4, w/4] => [b, 512, h/4, w/4]
        self.blk3 = CSAC_cell(512, ch_out, 3, 2)
        # # [b, 512, h/4, w/4] => [b,-1]
        self.outlayer = nn.Linear(ch_out*2*7, 6)  # parameters will be changed

    def forward(self, x):

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x

class CSAC_NET_v2(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(CSAC_NET_v2, self).__init__()

        # followed 3 blocks
        # [b, 128, h, w] => [b, 256, h/2 ,w/2]
        self.blk1 = CSAC_cell(ch_in, 64, 3, 2)
        # [b, 256, h/2, w/2] => [b, 512, h/4, w/4]
        self.blk2 = CSAC_cell(64, 128, 3, 2)
        # [b, 256, h/2, w/2] => [b, 512, h/4, w/4]
        self.blk3 = CSAC_cell(128, 256, 3, 2)
        # # [b, 512, h/4, w/4] => [b, 512, h/4, w/4]
        self.blk4 = CSAC_cell(256, ch_out, 2, 2)
        # # [b, 512, h/4, w/4] => [b,-1]
        self.outlayer = nn.Linear(ch_out*2*2, 3)  # parameters will be changed

    def forward(self, x):

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


# Linear NET    ************    Linear NET    ************      Linear NET    ************
class LINEAR_NET(nn.Module):
    def __init__(self, ch_in, ch_out, hidden_size1=256, hidden_size2=512):
        super(LINEAR_NET, self).__init__()

        self.fc1 = nn.Linear(ch_in * 16 * 63, hidden_size1)  # Flatten the input tensor
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, ch_out)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ResNET    ************    ResNET    ************      ResNET    ************
class ResBlk(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        # we add stride support for resbok, which is distinct from tutorials.
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        """
        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out
        out = F.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h ,w]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(128, 256, stride=2)
        # # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)
        # # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4 = ResBlk(512, 512, stride=1)

        self.outlayer = nn.Linear(512*1*1, 3)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print('after conv:', x.shape) #[b, 512, 2, 2]
        # [b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x

class ResNet18_udlsrn(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b, 64, h, w] => [b, 128, h ,w]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(128, 256, stride=2)
        # # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)
        # # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4 = ResBlk(512, 512, stride=1)

        self.outlayer = nn.Linear(512*1*1, 6)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print('after conv:', x.shape) #[b, 512, 2, 2]
        # [b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x


# LeNET5    ************    LeNET5    ************      LeNET5    ************
class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, 16, ]
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            #
        )
        # flatten
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(4608, 32),
            nn.ReLU(),
            # nn.Linear(120, 84),
            # nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        """
        :param x: [b, 3, 32, 32]
        :return:
        """
        batchsz = x.size(0)
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)
        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(batchsz, 4608)
        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)

        # # [b, 10]
        # pred = F.softmax(logits, dim=1)
        # loss = self.criteon(logits, y)

        return logits

# SENET    ************    SENET    ************      SENET    ************

class Block(nn.Module):
    def __init__(self, in_channels, filters, stride=1, is_1x1conv=False):
        super(Block, self).__init__()
        filter1, filter2, filter3 = filters
        self.is_1x1conv = is_1x1conv
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filter3))
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filter3))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(filter3, filter3 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(filter3 // 16, filter3, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        x_shortcut = x
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.se(x1)
        x1 = x1 * x2
        if self.is_1x1conv:
            x_shortcut = self.shortcut(x_shortcut)
        x1 = x1 + x_shortcut
        x1 = self.relu(x1)
        return x1


class SENet(nn.Module):
    def __init__(self, cfg):
        super(SENet, self).__init__()
        classes = cfg['classes']
        num = cfg['num']
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.conv2 = self._make_layer(64, (64, 64, 256), num[0], 1)
        self.conv3 = self._make_layer(256, (128, 128, 512), num[1], 2)
        self.conv4 = self._make_layer(512, (256, 256, 1024), num[2], 2)
        self.conv5 = self._make_layer(1024, (512, 512, 2048), num[3], 2)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(2048, classes))

    def _make_layer(self, in_channels, filters, num, stride=1):
        layers = []
        block_1 = Block(in_channels, filters, stride=stride, is_1x1conv=True)
        layers.append(block_1)
        for i in range(1, num):
            layers.append(Block(filters[2], filters, stride=1, is_1x1conv=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def Senet():
    cfg ={
        'num':(3, 4, 6, 3),
        'classes': (3)
    }
    return SENet(cfg)


# x = torch.randn(64, 8, 16, 63)
# net1 = Senet()
# y = net1.forward(x)
# print(y)
