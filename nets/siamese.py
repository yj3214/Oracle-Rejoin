import torch
import torch.nn as nn

from nets.ConvNeXt import convnext_base
from nets.MobileNet_ca import mbv2_ca, CoordAtt
from nets.resnet_50 import Bottleneck,ResNet,resnet50
from nets.shuffletnetv2 import shufflenet_v2
from nets.vgg import  VGG11
from nets.AlexNet import AlexNet
from nets.repvgg import RepVGG, RepVGG_B2
from collections import OrderedDict
from MBConvBlock import MBConv
from nets.mbv2 import mobilenet_v2
from nets.mbv2 import MobileNetV2


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height) 


class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()
        self.vgg = VGG11(pretrained, input_shape[-1])
        del self.vgg.avgpool
        del self.vgg.classifier
        self.MBConv = MBConv(in_channels=6, out_channels=3, drop_path=0.25)
        # self.MBConv2 = MBConv(in_channels=3, out_channels=3, drop_path=0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        flat_shape = 512 * 3 * 3
        self.classifier = nn.Sequential(
            nn.Linear(flat_shape, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )
        # self.conv = nn.Conv2d(6, 3, 1)
        # self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        # self.fully_connect2 = torch.nn.Linear(1000, 1)
    def forward(self, x):
        x1, x2 = x
        y = torch.cat((x1, x2), dim=1)
        # y = self.conv(y)
        #------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        #------------------------------------------#
        y = self.MBConv(y)
        # x1 = self.MBConv2(x1)
        # x2 = self.MBConv2(x2)
        # y = y + 0.05*(x1 + x2)
        #--------------------------------------------#
        y = self.vgg.features(y)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        # x1 = self.vgg.features(x1)
        # x2 = self.vgg.features(x2)
        #-------------------------#
        #   相减取绝对值
        #-------------------------#     
        # x1 = torch.flatten(x1, 1)
        # x2 = torch.flatten(x2, 1)
        # x = torch.abs(x1 - x2)
        #-------------------------#
        #   进行两次全连接
        #-------------------------#
        # x = self.fully_connect1(x)
        # x = self.fully_connect2(x)
        return y.unsqueeze(1)



def conv_bn_relu(in_channels, out_channels, kernel_size,  stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)),
        ('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
    ]))


def linear_bn_relu(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features, bias=False)),  # Bias is added after BN
        ('bn', nn.BatchNorm1d(out_features)),
        ('relu', nn.ReLU()),
    ]))
class SigNet(nn.Module):
    '''
    原始的signet
    '''
    """ SigNet model, from https://arxiv.org/abs/1705.05787
    """
    def __init__(self):
        super(SigNet, self).__init__()

        self.feature_space_size = 2048

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_relu(3, 96, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', conv_bn_relu(96, 256, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', conv_bn_relu(256, 384, 3, pad=1)),
            ('conv4', conv_bn_relu(384, 384, 3, pad=1)),
            ('conv5', conv_bn_relu(384, 256, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

    def forward(self, inputs):
        x = self.conv_layers(inputs)
        return x
class SigNet2(nn.Module):
    '''
    改进的signet
    '''
    def __init__(self):
        super(SigNet2, self).__init__()

        self.feature_space_size = 2048

        self.conv_layers = nn.Sequential(OrderedDict([
            ('conv1', conv_bn_relu(3, 64, 11, stride=4)),
            ('maxpool1', nn.MaxPool2d(3, 2)),
            ('conv2', conv_bn_relu(64, 256, 5, pad=2)),
            ('maxpool2', nn.MaxPool2d(3, 2)),
            ('conv3', conv_bn_relu(256, 512, 3, pad=1)),
            ('conv4', conv_bn_relu(512, 512, 3, pad=1)),
            ('conv5', conv_bn_relu(512, 512, 3, pad=1)),
            ('maxpool3', nn.MaxPool2d(3, 2)),
        ]))

    def forward(self, inputs):
        x = self.conv_layers(inputs)
        return x

class SiameseSignet(nn.Module):
    def __init__(self):
        super(SiameseSignet, self).__init__()
        self.signet = SigNet2()
        # self.vgg = VGG16(pretrained, input_shape[-1])
        # del self.vgg.avgpool
        # del self.vgg.classifier
        
        # flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(2048, 2048)
        self.fully_connect2 = torch.nn.Linear(2048, 1)

    def forward(self, x):
        x1, x2 = x
        #------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        #------------------------------------------#
        x1 = self.signet(x1)
        x2 = self.signet(x2)
        # x1 = self.vgg.features(x1)
        # x2 = self.vgg.features(x2)   
        #-------------------------#
        #   相减取绝对值
        #-------------------------#     
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        #-------------------------#
        #   进行两次全连接
        #-------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x


class Siamese_resnet50(nn.Module):
    def __init__(self):
        super(Siamese_resnet50, self).__init__()
        # self.signet = SigNet2()
        # self.resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        self.resnet50 = resnet50()
        # del self.resnet50.avgpool
        # del self.resnet50.classifier
        # self.vgg = VGG16(pretrained, input_shape[-1])
        # del self.vgg.avgpool
        # del self.vgg.classifier
        self.MBConv1 = MBConv(in_channels=6, out_channels=3, drop_path=0.25)
        self.MBConv2 = MBConv(in_channels=3, out_channels=3, drop_path=0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )
        # flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        # self.fully_connect1 = torch.nn.Linear(2048, 2048)
        # self.fully_connect2 = torch.nn.Linear(2048, 1)

    def forward(self, x):
        x1, x2 = x
        y = torch.cat((x1, x2), dim=1)
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        y = self.MBConv1(y)
        x1 = self.MBConv2(x1)
        x2 = self.MBConv2(x2)
        y = y + x1 + x2
        y = self.resnet50(y)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        return y


class Siamese_Alexnet(nn.Module):
    def __init__(self):
        super(Siamese_Alexnet, self).__init__()
        # self.signet = SigNet2()

        self.alexnet = AlexNet()
        # self.vgg = VGG16(pretrained, input_shape[-1])
        # del self.vgg.avgpool
        # del self.vgg.classifier
        self.MBConv1 = MBConv(in_channels=6, out_channels=3, drop_path=0.25)
        self.MBConv2 = MBConv(in_channels=3, out_channels=3, drop_path=0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )
        # flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        # self.fully_connect1 = torch.nn.Linear(1024, 1024)
        # self.fully_connect2 = torch.nn.Linear(1024, 1)

    def forward(self, x):
        x1, x2 = x
        y = torch.cat((x1, x2), dim=1)
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        y = self.MBConv1(y)
        x1 = self.MBConv2(x1)
        x2 = self.MBConv2(x2)
        y = y + x1 + x2
        y = self.alexnet.features(y)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        return y.unsqueeze(1)



class Siamese_RepVGG_B2(nn.Module):
    def __init__(self):
        super(Siamese_RepVGG_B2, self).__init__()
        # self.signet = SigNet2()
        self.RepVGG_B2 = RepVGG_B2()

        # self.alexnet = AlexNet()
        # self.vgg = VGG16(pretrained, input_shape[-1])
        # del self.vgg.avgpool
        # del self.vgg.classifier

        # flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(2560, 2560)
        self.fully_connect2 = torch.nn.Linear(2560, 1)

    def forward(self, x):
        x1, x2 = x

        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        x1 = self.RepVGG_B2(x1)
        x2 = self.RepVGG_B2(x2)
        # x1 = self.vgg.features(x1)
        # x2 = self.vgg.features(x2)
        # -------------------------#
        #   相减取绝对值
        # -------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        # -------------------------#
        #   进行两次全连接
        # -------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x

class Siamese_2ch_RepVGG_B2(nn.Module):
    def __init__(self):
        super(Siamese_2ch_RepVGG_B2, self).__init__()
        # self.signet = SigNet2()
        self.RepVGG_B2 = RepVGG_B2()
        self.MBConv = MBConv(in_channels=6,out_channels=3,drop_path=0.25)

        # self.alexnet = AlexNet()
        # self.vgg = VGG16(pretrained, input_shape[-1])
        # del self.vgg.avgpool
        # del self.vgg.classifier

        # flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(2560, 2560)
        self.fully_connect2 = torch.nn.Linear(2560, 1)

    def forward(self, x):
        x1, x2 = x
        x_2ch = torch.cat((x1,x2),dim=1)
        # 经过MBConv
        x_2ch = self.MBConv(x_2ch)
        x_2ch = self.RepVGG_B2(x_2ch)
        x_2ch = torch.flatten(x_2ch, 1)
        # -------------------------#
        #   进行两次全连接
        # -------------------------#
        x_2ch = self.fully_connect1(x_2ch)
        x_2ch = self.fully_connect2(x_2ch)
        return x_2ch




class Siamese_mbv2(nn.Module):
    def __init__(self):
        super(Siamese_mbv2, self).__init__()
        # self.signet = SigNet2()
        self.mbv2 = mobilenet_v2()
        self.MBConv = MBConv(in_channels=6, out_channels=3, drop_path=0.25)
        self.MBConv2 = MBConv(in_channels=3, out_channels=3, drop_path=0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        flat_shape = 1280 * 1 * 1
        self.classifier = nn.Sequential(
            # nn.Linear(flat_shape, 4096),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Linear(flat_shape, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x1, x2 = x
        y = torch.cat((x1, x2), dim=1)
        # y = self.conv(y)
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        y = self.MBConv(y)
        x1 = self.MBConv2(x1)
        x2 = self.MBConv2(x2)
        y = y + 0.05*(x1 + x2)
        # --------------------------------------------#
        y = self.mbv2(y)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        return y



class Siamese_shv2(nn.Module):
    def __init__(self):
        super(Siamese_shv2, self).__init__()
        self.shv2 = shufflenet_v2()
        self.MBConv = MBConv(in_channels=6, out_channels=3, drop_path=0.25)
        self.MBConv2 = MBConv(in_channels=3, out_channels=3, drop_path=0.25)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        flat_shape = 1024
        self.classifier = nn.Sequential(
            # nn.Linear(flat_shape, 4096),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Linear(flat_shape, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x1, x2 = x
        y = torch.cat((x1, x2), dim=1)
        # y = self.conv(y)
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        y = self.MBConv(y)
        x1 = self.MBConv2(x1)
        x2 = self.MBConv2(x2)
        y = y + 0.05*(x1 + x2)
        # --------------------------------------------#
        y = self.shv2(y)
        # y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        return y