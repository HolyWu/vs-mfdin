import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class BasicConv(nn.Module):
    def __init__(self, nf=64):
        super(BasicConv, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)  # 0.1的比例初始化权重

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return out


class RFABlock(nn.Module):
    def __init__(self, nf=64):
        super(RFABlock, self).__init__()
        self.res1 = BasicConv(nf)
        self.res2 = BasicConv(nf)
        self.res3 = BasicConv(nf)
        self.res4 = BasicConv(nf)
        self.conv = nn.Conv2d(nf * 4, nf, 1, bias=False)

    def forward(self, x):
        identity = x
        fea1 = self.res1(x)
        xin2 = identity + fea1
        fea2 = self.res2(xin2)
        xin3 = xin2 + fea2
        fea3 = self.res3(xin3)
        xin4 = xin3 + fea3
        fea4 = self.res4(xin4)
        out = self.conv(torch.cat([fea1, fea2, fea3, fea4], dim=1))

        return identity + out


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)  # 0.1的比例初始化权重

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out
