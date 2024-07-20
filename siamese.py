import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch import nn
from nets.siamese import Siamese as siamese, Siamese_RepVGG_B2, Siamese_2ch_RepVGG_B2, Siamese_mbv2, Siamese_shv2
from nets.siamese import Siamese_resnet50
from nets.siamese import Siamese_Alexnet

#---------------------------------------------------#
#   使用自己训练好的模型预测需要修改model_path参数
#---------------------------------------------------#
class Siamese(object):
    _defaults_resnet50 = {
        # "model_path"    : 'logs/output2/ep002-loss0.032-val_loss0.046.pth',
        # "model_path"    : 'logs/ep011-loss0.007-val_loss0.015.pth',
        "model_path"    : './without_expend_logs/siamese/resnet50/ep078-loss0.0002-val_loss0.3905.pth',

        "input_shape"   : (105, 105, 3),
        "cuda"          : True
    }
    _defaults_shv2 = {
    "model_path": './without_expend_logs/siamese/shv2/ep090-loss0.0045-val_loss0.2784.pth',
    "input_shape": (105, 105, 3),
    "cuda": True

    }
    _defaults_vgg16 = {
        #-------------2ch-----------------------#
        # "model_path": './without_expend_logs/siamese/vgg16/ep087-loss0.0032-val_loss0.2016.pth',
        #----------------VGG16-------------------------#
        "model_path": './expend_logs/expend_lines_50000/vgg16/ep021-loss0.0046-val_loss0.0051.pth',

        # "model_path"    : './without_expend_logs/siamese/vgg16/ep084-loss0.0001-val_loss0.5937.pth',
        # "model_path"    : './expend_logs/expend_lines_5000/vgg16/ep086-loss0.0006-val_loss0.0039.pth',
        # "model_path"    : './expend_logs/expend_lines_10000/vgg16/ep100-loss0.004-val_loss0.408.pth',
        # "model_path"     : './expend_logs/expend_lines_15000/vgg16/ep061-loss0.0008-val_loss0.0076.pth',
        # "model_path"    : './expend_logs/expend_lines_15000/vgg16/ep100-loss0.0010-val_loss0.0356.pth',
        # "model_path"    : './expend_logs/expend_lines_25000/vgg16/ep099-loss0.0009-val_loss0.0181.pth',
        # "model_path"    : './expend_logs/expend_lines_35000/vgg16/ep047-loss0.0016-val_loss0.0049.pth',
        # "model_path": './expend_logs/expend_lines_35000/vgg16/ep068-loss0.0010-val_loss0.0096.pth',

        # "model_path"    : './without_expend_logs/siamese/vgg16/ep055-loss0.0009-val_loss0.2101.pth',  # alpha = 0.1
        # "model_path": './without_expend_logs/siamese/vgg16/ep060-loss0.0003-val_loss0.5747.pth',  # alpha = 0.25
        # "model_path": './without_expend_logs/siamese/vgg16/ep079-loss0.0002-val_loss1.2873.pth',  # alpha = 0.5
        # "model_path": './without_expend_logs/siamese/vgg16/ep059-loss0.0003-val_loss1.5478.pth',  # alpha = 0.75
        # "model_path": './without_expend_logs/siamese/vgg16/ep085-loss0.0001-val_loss1.7972.pth',  # alpha = 0.9
        # "model_path": './without_expend_logs/siamese/vgg16/ep091-loss0.0001-val_loss0.4443.pth',  # alpha = 0.5 gamma = 1
        # "model_path": './without_expend_logs/siamese/vgg16/ep099-loss0.0001-val_loss0.5948.pth',  # alpha = 0.5 gamma = 1.25
        # "model_path": './without_expend_logs/siamese/vgg16/ep073-loss0.0001-val_loss0.3166.pth', # alpha = 0.5 gamma = 1.5
        # "model_path": './without_expend_logs/siamese/vgg16/ep056-loss0.0001-val_loss0.1732.pth', # alpha = 0.25 gamma = 1.75
        # "model_path": './without_expend_logs/siamese/vgg16/ep060-loss0.0002-val_loss0.2862.pth', # alpha = 0.25 gamma = 1.5
        # "model_path": './without_expend_logs/siamese/vgg16/ep047-loss0.0003-val_loss0.1330.pth', # alpha = 0.1 gamma = 2
        # "model_path": './without_expend_logs/siamese/vgg16/ep061-loss0.0001-val_loss0.0435.pth', # alpha = 0.1 gamma = 3
        # "model_path": './without_expend_logs/siamese/vgg16/ep075-loss0.0001-val_loss0.0261.pth', # alpha = 0.1 gamma = 5
        # "model_path": './without_expend_logs/siamese/vgg16/ep045-loss0.0002-val_loss0.8545.pth', # alpha = 0.75 gamma = 1
        # "model_path": './without_expend_logs/siamese/vgg16/ep049-loss0.0002-val_loss0.3184.pth',# alpha = 0.75 gamma = 1


        # "model_path"    : 'logs_resnet50/ep011-loss0.014-val_loss0.114.pth',
        "input_shape"   : (105, 105, 3),
        "cuda"          : True
    }
    _defaults_alexnet = {
        # "model_path"    : 'logs/output2/ep002-loss0.032-val_loss0.046.pth',
        "model_path"    : './without_expend_logs/siamese/alexnet/ep080-loss0.0088-val_loss0.2067.pth',
        # "model_path"    : 'logs_resnet50/ep011-loss0.014-val_loss0.114.pth',
        "input_shape"   : (105, 105, 3),
        "cuda"          : True
    }

    _defaults_mbv2 = {
        "model_path": './without_expend_logs/siamese/mbv2/ep095-loss0.0076-val_loss0.0856.pth',
        "input_shape": (105, 105, 3),
        "cuda": True
    }



    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Siamese
    #---------------------------------------------------#
    def __init__(self,flag, **kwargs):
        self.flag = flag
        if flag == 'resnet50':
            self.__dict__.update(self._defaults_resnet50)
        elif flag == 'vgg16':
            self.__dict__.update(self._defaults_vgg16)
        elif flag == 'mbv2':
            self.__dict__.update(self._defaults_mbv2)
        elif flag == 'shv2':
            self.__dict__.update(self._defaults_shv2)
        else:
            self.__dict__.update(self._defaults_alexnet)
        self.generate()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        #---------------------------#
        #   载入模型与权值
        #---------------------------#
        print('Loading weights into state dict...')
        device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.flag == 'resnet50':
            model = Siamese_resnet50()
        elif self.flag == 'mbv2':
            model = Siamese_mbv2()
        elif self.flag == 'shv2':
            model = Siamese_shv2()
        elif self.flag == 'vgg16': 
            model = siamese(self.input_shape)
        else:
            model = Siamese_Alexnet()
        # model = model()
        model.load_state_dict(state_dict=torch.load(self.model_path, map_location=device))
        self.net = model.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
    
    def letterbox_image(self, image, size):
        image   = image.convert("RGB")
        iw, ih  = image.size
        w, h    = size
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image       = image.resize((nw,nh), Image.BICUBIC)
        new_image   = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        if self.input_shape[-1]==1:
            new_image = new_image.convert("L")
        return new_image
        
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        #---------------------------------------------------#
        #   对输入图像进行不失真的resize
        #---------------------------------------------------#
        image_1 = self.letterbox_image(image_1,[self.input_shape[1],self.input_shape[0]])
        image_2 = self.letterbox_image(image_2,[self.input_shape[1],self.input_shape[0]])
        
        #---------------------------------------------------#
        #   对输入图像进行归一化
        #---------------------------------------------------#
        photo_1 = np.asarray(image_1).astype(np.float64) / 255
        photo_2 = np.asarray(image_2).astype(np.float64) / 255

        if self.input_shape[-1]==1:
            photo_1 = np.expand_dims(photo_1, -1)
            photo_2 = np.expand_dims(photo_2, -1)

        with torch.no_grad():
            #---------------------------------------------------#
            #   添加上batch维度，才可以放入网络中预测
            #---------------------------------------------------#
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(photo_1, (2, 0, 1)), 0)).type(torch.FloatTensor)
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(photo_2, (2, 0, 1)), 0)).type(torch.FloatTensor)
            
            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()
                
            #---------------------------------------------------#
            #   获得预测结果，output输出为概率
            #---------------------------------------------------#
            output = self.net([photo_1, photo_2])[0]
            output = torch.nn.Sigmoid()(output)

        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(image_1))

        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(image_2))
        # plt.text(-12, -12, 'Similarity:%.3f' % output, ha='center', va= 'bottom',fontsize=11)
        # plt.show()
        return output
