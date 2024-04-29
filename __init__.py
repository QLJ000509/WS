from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from network.resnet import *
from network.msc import *


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def ResNet101(n_classes):
    return ResNet(n_classes=n_classes, n_blocks=[3, 4, 23, 3])




    for name, module in base.named_modules():
        if ".bn" in name:
            module.momentum = 0.9997

    return MSC(base=base, scales=[0.5, 0.75])

