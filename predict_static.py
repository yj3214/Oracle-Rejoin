from cProfile import label
from numpy import ma
from pip import main
import argparse

def read_dict(dict_name):
    f = open(dict_name, 'r')
    a = f.read()
    dict = eval(a)
    f.close()
    return dict

def staACC(threshold,dic):
    '''

    dict = {'up_file':{‘down_file’:[index,score]}}

    '''
    #便利dic的key
    if len(dic) != 158:
        print('warning!!!!   not 158!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return 9999999999999999
    yes = 0         #up 和 down配对的排名 在 threshold 之前 
    no = 0
    all_num = len(dic)     #是 up文件夹中的数目
    for each_up in dic.keys():

        #根据 each_up的文件名 确定和他匹配的 down的文件名

        
        if each_up.split('_')[1] == 'up.jpg':

            aim_down_file_name = each_up.split('_')[0] + '_down.jpg'
        else:
            aim_down_file_name = each_up.split('_')[0] + '_up.jpg'
        # 找出aim_down_file_name对应的置信度和排名
        # aaa = dic[each_up]
        # i = 0
        # for s in aaa:
        #     i+=1
        #     if i> 800:
        #         a = 444
        #     print(s)
            # print(dic[each_up][s][0])
            # print(dic[each_up][s][1])
        # print(dic[each_up])
        aim_index, aimscore  = dic[each_up][aim_down_file_name]

        # print("{} and {}: index--{}  score--{}".format(each_up,aim_down_file_name,aim_index,aimscore) )

        #判断 是否成功匹配  判断条件：
        #   （1）排名在前threshold内
        if aim_index <= threshold:

            # print(aim_index)
            #成功匹配
            yes += 1
        else:
            #匹配失败
            no += 1

    acc = yes/all_num 
    # print(all_num)
    # print(no)
    # print(yes)
    return acc
        # print(each_up)
def ErrorRateAt95Recall1(labels, scores):
    import numpy as np
    labels =np.array(labels)
    scores = np.array(scores)
    recall_point = 0.95
    #对分数（0-1）降序排序
    # indices = np.argsort(scores)[::-1]    #降序排列
    indices = np.argsort(scores)    #降序排列
    sorted_labels = labels[indices]
    sorted_scores = scores[indices]
    #
    n_match = sum(sorted_labels)
    n_thresh = recall_point * n_match
    # a = np.cumsum(sorted_labels)
    # b = np.cumsum(sorted_labels) >= n_thresh

    thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
    FP = np.sum(sorted_labels[:thresh_index] == 0)
    TN = np.sum(sorted_labels[thresh_index:] == 0)
    return float(FP) / float(FP + TN)
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dict_path', help='dict path', default='./TXT/expend_50000/siamese_vgg11_test2_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/siamese/siamese_vgg11_old_epoch55_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/siamese/alexnet/siamese_alexnet_new_output1_epoch80_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/siamese/resnet50/siamese_resnet50_new_output1_epoch78_without.txt')
    # argparser.add_argument('--dict_path', help='dict path', default='./TXT/without_expend/siamese/vgg16/siamese_vgg16_output1_epoch85_without.txt')

    # -----------------------------Focal Loss 调参 ---------------------------------- #
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.1_output1_epoch92_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.75_gamma=1_epoch40_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.5_output1_epoch78_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.75_output1_epoch59_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.9_output1_epoch85_without.txt')
    # argparser.add_argument('--dict_path', help='dict path', default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.5_gamma=1_epoch91_without.txt')
    # argparser.add_argument('--dict_path', help='dict path', default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.5_gamma=1.25_epoch99_without.txt')
    # argparser.add_argument('--dict_path', help='dict path', default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.5_gamma=1.5_epoch73_without.txt')
    # argparser.add_argument('--dict_path', help='dict path', default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.25_gamma=1.75_epoch56_without.txt')
    # argparser.add_argument('--dict_path', help='dict path', default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.25_gamma=1.5_epoch59_without.txt')
    # argparser.add_argument('--dict_path', help='dict path', default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.1_gamma=2_epoch47_without.txt')
    # argparser.add_argument('--dict_path', help='dict path', default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.1_gamma=3_epoch61_without.txt')
    # argparser.add_argument('--dict_path', help='dict path', default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_alpha=0.1_gamma=5_epoch75_without.txt')
    # argparser.add_argument('--dict_path', help='dict path', default='./TXT/without_expend/siamese/vgg11/siamese_vgg11bn_alpha=0.25_gamma=1.0_mix_epoch49_without.txt')
    # --------------------------------------ConvNext-----------------#
    # argparser.add_argument('--dict_path', help='dict path', default='./TXT/without_expend/siamese/ch_RepVGG/siamese_ch_RepVGG_B2_FocalLoss_gama1_output1_epoch098_without.txt')

#------------------------------------------------------------------------------#
#加干扰
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_shortcut_output1_epoch84.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/siamese/alexnet/siamese_alexnet_new_output1_epoch80.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/siamese/resnet50/siamese_resnet50_new_output1_epoch78.txt')

    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/siamese/vgg16/siamese_vgg16bn_a0.75g1_epoch45.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/triple/vgg16/triple_vgg16_output2_epoch93.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/without_expend/triple/resnet50/triple_resnet50_output2_epoch98.txt')
    # argparser.add_argument('--dict_path', help='dict path', default='./TXT/without_expend/siamese/CIR/siamese_RepVGG_B2_DiceLoss_output1_epoch94.txt')
    # argparser.add_argument('--dict_path', help='dict path', default= './TXT/expend_35000/siamese_vgg11_output1_epoch66_new.txt')
#------------------------------------------------------------------------------#
#EXPEND------VGG16
#    ----------------------siamese
#     argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_5000/siamese_vgg11_new_a0.25g1_epoch86_without_best.txt')
#     argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_5000/siamese_vgg16_new_a0.75g1_epoch54.txt')

    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_10000/siamese_RepVGG_B2_FocalLoss_gama1_output1_epoch096_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_10000/siamese_vgg16_output1_epoch100.txt')

    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_15000/siamese_vgg11_output1_newest_epoch100_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_15000/siamese_vgg11_output1_newest_epoch100.txt')
    
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_20000/siamese_vgg16_output1_epoch50_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_20000/siamese_vgg16_output1_epoch50.txt')
    
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_25000/siamese_vgg11_output1_newest_epoch99_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_25000/siamese_vgg11_output1_newest_epoch99.txt')
#    ----------------------triple
#     argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_5000/triple_resnet50_output2_epoch94_pearson_without.txt')
#     argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_35000/siamese_vgg16_output1_new_epoch54_without.txt')

    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_10000/triple_resnet50_output2_epoch100_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_10000/triple_resnet50_output2_epoch30_without.txt')

    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_15000/triple_resnet50_output2_epoch100_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_35000/siamese_vgg16_output1_epoch51_ago_best_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_35000/siamese_vgg16_output1_epoch11_ago_best_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_35000/siamese_vgg11_new_epoch47_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='./TXT/expend_15000/siamese_vgg11bn_output1_new_epoch50_without.txt')
    # argparser.add_argument('--dict_path', help='dict path',default='G:/1########另存代码/Siamese-pytorch-master/logs/triple_network_vgg_ep060-loss0.012-val_loss0.019.pth_predict_minedata_without_inter.txt')




    # argparser.add_argument('--dict_path', help='dict path',default='img148.txt')
    argparser.add_argument('--staACC', help='1:statistic acc 0: not statistic acc' ,default=1)
    # argparser.add_argument('--th', help='threshould value' ,default=10)
    args = argparser.parse_args()
    # th = [1,3,5,10,15,20,25,30,35,45,50]
    # th = [1,3,5,10,15,20,25,30,35,40,45,50,55,60,65,70]
    th = [1, 3, 5, 10, 20, 30, 50]
    # th = [1]

    ####
    ####        all_up_dict = {'up_file':{‘down_file’:[index,score]}}
    ####
    all_up_dict = read_dict(args.dict_path)
    labels = []
    scores = []
    for each in all_up_dict:
        # print(each)
        if each.split('_')[1] == 'up.jpg':

            aim_down_file_name = each.split('_')[0] + '_down.jpg'
        else:
            aim_down_file_name = each.split('_')[0] + '_up.jpg'
        for se_each in all_up_dict[each]:
            if se_each == aim_down_file_name:
                la = 1
            else:
                la = 0
            labels.append(la)
            scores.append(all_up_dict[each][se_each][1])
            # print(se_each)
            # print(all_up_dict[each][se_each][1])
    from scipy import interpolate
    from sklearn.metrics import roc_curve,auc
    print(ErrorRateAt95Recall1(labels,scores))
    fpr,tpr,thresh = roc_curve(labels,scores)
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
    print('FPR95:', fpr95)
    # print(len(labels))

        


    for t in th:
        if args.staACC:
            acc = staACC(t, all_up_dict)
            # print('threshold {} is :{}'.format(t,acc) )
            print(acc)
            # print(acc)




    # print('debug')

