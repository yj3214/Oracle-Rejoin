from ast import arg
from operator import index
from pickle import TRUE
from xml.dom.minicompat import defproperty
from matplotlib import image
import numpy as np
from PIL import Image
import cv2
from torch import mode
from siamese import Siamese
import os
import argparse
from tqdm import *
'''
调用模型预测
1、真实配对的图片的分值是多少  ---- 用于寻找阈值
2、统计曲线a 与 其他曲线的分值 统计出来看一下分布情况-----确定评价指标（精准率和召回率还有F1-score）


两个文件夹A B
step1. 
    for a in A:
        a与B中每一条曲线计算拟合度
        拟合度排序



'''

def cv2_img(img):
    img = cv2.imread(img)
    
    height, width,_ = img.shape

    if height > width:
        reimg = cv2.copyMakeBorder(img,0,0,(height - width)//2,(height - width)//2,cv2.BORDER_CONSTANT, value=[255,255,255])
        reimg = cv2.resize(reimg,(105,105))
    else:
        reimg = cv2.copyMakeBorder(img,(width - height)//2,(width - height)//2,0,0,cv2.BORDER_CONSTANT, value=[255,255,255])
        reimg = cv2.resize(reimg,(105,105))

    img_nummpy = np.array(reimg)
    image = Image.fromarray(img_nummpy.astype('uint8')).convert('RGB')
    return image
# def pre_img(path):

def re_lis(z):
    '''
    form [file,score],index to {file:[index,score]}
    '''
    dic = {}
    for i,index in z:     #i[0]:  file  i[1]:   score
        dic[i[0]]=[index,i[1]]
    return dic

    
def saveTxt(ret, path):
    f = open(path, 'w')
    f.write(str(ret))
    f.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root_path', help='the root path',default='./img/')
    argparser.add_argument('--saveTXT', help='1:save 0: not save',default=1)
    argparser.add_argument('--inter_img_path', help='gan rao img',default='D:/fighting/5_join_together/dataset/join_together/Original_unknown_1083')
    argparser.add_argument('--inter_flag', help='inter_flag',default=0)   # 原来为0  0表示不加干扰 1表示加上干扰项

    vgg16 = 1
    mbv2 = 0
    resnet50 = 0
    alexnet = 0
    shv2 = 0

    
    args = argparser.parse_args()
    # argparser = argparser.parse_args()
    if vgg16:
        model = Siamese('vgg16')
    elif resnet50:
        model = Siamese('resnet50')
    elif mbv2:
        model = Siamese('mbv2')
    elif shv2:
        model = Siamese('shv2')
    else:
        model = Siamese('alexnet')
    # model = Siamese()       #加载模型

    up_file_path = os.path.join(args.root_path, 'img')
    down_file_path = os.path.join(args.root_path, 'img')
    all_up = {}

    for file_up in tqdm(os.listdir(up_file_path)):
        dic_name = 'dic_up_' + file_up
        # dic_up = dict()
        up = []

        img_up_path = os.path.join(up_file_path,file_up)
        img_up = cv2_img(img_up_path)

        # down = os.listdir(down_file_path)
        # 加入干扰项

        # inter_down = os.listdir(args.inter_img_path)
        for file_down in os.listdir(down_file_path):
            if file_up == file_down:
                continue

            img_down_path = os.path.join(down_file_path,file_down)
            img_down = cv2_img(img_down_path)

            probability = round(model.detect_image(img_up,img_down ).item(), 4)

            up.append([file_down,probability])
        
        if args.inter_flag:
            for file_down in os.listdir(args.inter_img_path):

                img_down_path = os.path.join(args.inter_img_path,file_down)
                img_down = cv2_img(img_down_path)

                probability = round(model.detect_image(img_up,img_down ).item(), 4)

                up.append([file_down,probability])

        up = sorted(up, key=lambda d:d[1], reverse = True)
        rate_index = [i for i in range (1, len(up) + 1)]

        up_dic = re_lis( zip(up,rate_index))        #{file:[index,score]}   

        all_up[file_up] = up_dic        # up：up_dic
        # print(len(all_up))
    if args.saveTXT:                #
        if vgg16:
            if args.inter_flag:
                 # saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg11bn_backbone_2ch'+ '.txt')    # 85
                # saveTxt(all_up, './TXT/expend_5000/siamese_vgg11_new_a0.25g1_epoch86'+ '.txt')
                # saveTxt(all_up, './TXT/expend_10000/siamese_vgg16_output1_epoch100'+ '.txt')
                # saveTxt(all_up, './TXT/expend_15000/siamese_vgg11_output1_newest_epoch100'+ '.txt')
                # saveTxt(all_up, './TXT/expend_35000/siamese_vgg11_output1_epoch51_ago_best'+ '.txt')
                # saveTxt(all_up, './TXT/expend_20000/siamese_vgg16_output1_epoch50'+ '.txt')
                saveTxt(all_up, './TXT/expend_50000/siamese_vgg11_test1'+ '.txt')
                #  saveTxt(all_up, './TXT/expend_25000/siamese_vgg11_output1_newest_epoch99' + '.txt')
 
            else:
                #----------------------2ch-------------------
                # saveTxt(all_up, './TXT/without_expend/siamese/vgg11/siamese_vgg11bn_2ch_output1_epoch87' + '_without.txt')
                #----------------------vgg16----------------#
                # saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_backbone_epoch99' + '_without.txt')
                #--------------------------vgg11----------------#
                saveTxt(all_up, './TXT/expend_50000/siamese_vgg11_test2' + '_without.txt')
                 # saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_output1_epoch84'+ '_without.txt')
                # saveTxt(all_up, './TXT/expend_5000/siamese_vgg11_new_a0.25g1_epoch86'+ '_without.txt')
                # saveTxt(all_up, './TXT/expend_10000/siamese_vgg16_output1_epoch100'+ '_without.txt')
                # saveTxt(all_up, './TXT/expend_15000/siamese_vgg11_output1_newest_epoch100'+ '_without.txt')
                # saveTxt(all_up, './TXT/expend_35000/siamese_vgg16_output1_epoch51_ago_best'+ '_without.txt')
                # saveTxt(all_up, './TXT/expend_35000/siamese_vgg11_new_epoch47'+ '_without.txt')
                #  saveTxt(all_up, './TXT/without_expend/siamese/siamese_vgg11_old_epoch55' + '_without.txt')
                #  saveTxt(all_up, './TXT/expend_25000/siamese_vgg11_output1_newest_epoch99' + '_without.txt')

                # -------------------Focal Loss 调参---------------------------------#
                # saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.1_output1_epoch92'+ '_without.txt')
                #  saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.25_output1_epoch60' + '_without.txt')
                # saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.5_output1_epoch78' + '_without.txt')
                #  saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.75_output1_epoch59' + '_without.txt')
                # saveTxt(all_up,'./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.9_output1_epoch85' + '_without.txt')
                #  saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.5_gamma=1_epoch91' + '_without.txt')
                # saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.5_gamma=1.25_epoch99' + '_without.txt')
                # saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.5_gamma=1.5_epoch73' + '_without.txt')
                #  saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.25_gamma=1.75_epoch56' + '_without.txt')
                # saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.25_gamma=1.5_epoch59' + '_without.txt')
                #  saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.1_gamma=2_epoch47' + '_without.txt')
                # saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.1_gamma=3_epoch61' + '_without.txt')
                # saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.1_gamma=5_epoch75' + '_without.txt')
                # saveTxt(all_up, './TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.75_gamma=1_epoch45' + '_without.txt')
                #  saveTxt(all_up, './TXT/without_expend/siamese/vgg11/siamese_vgg11bn_alpha=0.25_gamma=1.0_mix_epoch49' + '_without.txt')

        if resnet50:

            if args.inter_flag:
                saveTxt(all_up, './TXT/without_expend/siamese/resnet50/siamese_resnet50_new_output1_epoch78'+ '.txt')
            else:
                saveTxt(all_up, './TXT/without_expend/siamese/resnet50/siamese_resnet50_new_output1_epoch78'+ '_without.txt')
        if alexnet:
            if args.inter_flag:
                saveTxt(all_up, './TXT/without_expend/siamese/alexnet/siamese_alexnet_new_output1_epoch80'+ '.txt')
            else:
                saveTxt(all_up, './TXT/without_expend/siamese/alexnet/siamese_alexnet_new_output1_epoch80'+ '_without.txt')
        if mbv2:
            if args.inter_flag:
                # saveTxt(all_up,'./TXT/without_expend/siamese/CIR/siamese_RepVGG_B2_output1_epoch100'+ '.txt')
                # saveTxt(all_up, './TXT/without_expend/siamese/CIR/siamese_RepVGG_B2_FocalLoss_output1_epoch100' + '.txt')
                # saveTxt(all_up, './TXT/without_expend/siamese/CIR/siamese_RepVGG_B2_DiceLoss_output1_epoch94' + '.txt')
                # saveTxt(all_up, './TXT/expend_10000/siamese_RepVGG_B2_DiceLoss_output1_epoch054' + '.txt')
                saveTxt(all_up, './TXT/without_expend/siamese/mbv2/siamese_mbv2_output1_epoch095' + '.txt')
            else:
                # saveTxt(all_up,'./TXT/without_expend/siamese/CIR/siamese_RepVGG_B2_output1_epoch100'+ '_without.txt')
                # saveTxt(all_up, './TXT/without_expend/siamese/CIR/siamese_RepVGG_B2_FocalLoss_output1_epoch96' + '_without.txt')
                # saveTxt(all_up, './TXT/without_expend/siamese/CIR/siamese_RepVGG_B2_DiceLoss_output1_epoch94' + '_without.txt')
                # saveTxt(all_up, './TXT/expend_15000/siamese_RepVGG_B2_FocalLoss_gama1_output1_epoch038' + '_without.txt')
                # saveTxt(all_up, './TXT/expend_5000/siamese_mbv2_output1_epoch70' + '_without.txt')
                saveTxt(all_up, './TXT/without_expend/siamese/mbv2/siamese_mbv2_output1_epoch095' + '_without.txt')
        if shv2:
            if args.inter_flag:
                saveTxt(all_up, './TXT/without_expend/siamese/shv2/siamese_shv2_output1_epoch96'+ '.txt')
            else:
                saveTxt(all_up, './TXT/without_expend/siamese/shv2/siamese_shv2_output1_epoch96'+ '_without.txt')




        # if args.staACC:
            # staACC(args.th,all_up)

        


    print('end')
