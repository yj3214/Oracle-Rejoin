import torch.nn as nn
import torch.nn.functional as F
from nets.resnet_50 import resnet50
import torch


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

class EmbeddingNet(nn.Module):
    def __init__(self,input_shape, pretrained=False):
        super(EmbeddingNet, self).__init__()
        self.resnet50 = resnet50()
        # self.vgg = VGG16(pretrained, input_shape[-1])
        # del self.vgg.avgpool
        # del self.vgg.classifier
         #flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(2048, 512)
        self.fully_connect2 = torch.nn.Linear(512, 2)

    def forward(self, x):
        output = self.resnet50(x)
        # output = self.vgg.features(x)
        output = torch.flatten(output, 1)
        output = self.fully_connect1(output)
        output = self.fully_connect2(output)
        # output = self.convnet(x)
        # output = output.view(output.size()[0], -1)
        # output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
    #     self.vgg = VGG16(pretrained, input_shape[-1])
    #     del self.vgg.avgpool
    #     del self.vgg.classifier
        
    #     flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
    #     self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
    #     self.fully_connect2 = torch.nn.Linear(512, 1)
    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
