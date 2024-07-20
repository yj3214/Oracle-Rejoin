import os
import random
from random import shuffle

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class TripletDataset(Dataset):
    def __init__(self, input_shape, dataset_path, num_train, num_val, train=True, train_own_data=True):
        super(TripletDataset, self).__init__()

        self.dataset_path   = dataset_path
        self.image_height   = input_shape[0]
        self.image_width    = input_shape[1]
        self.channel        = input_shape[2]
        
        self.train_lines    = []
        self.train_labels   = []

        self.val_lines      = []
        self.val_labels     = []
        self.types          = 0

        self.num_train      = num_train
        self.num_val        = num_val
        self.train          = train
        self.train_own_data = train_own_data
        self.load_dataset()

        mean, std = 0.1307, 0.3081
        self.transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(std=(std,),mean=(mean,))])

    def __len__(self):
        if self.train:
            return self.num_train
        else:
            return self.num_val

    def load_dataset(self):
        train_path = os.path.join(self.dataset_path, 'lines')
        if self.train_own_data:
            #-------------------------------------------------------------#
            #   自己的数据集，遍历大循环
            #-------------------------------------------------------------#
            for character in os.listdir(train_path):
                #-------------------------------------------------------------#
                #   对每张图片进行遍历
                #-------------------------------------------------------------#
                character_path = os.path.join(train_path, character)
                for image in os.listdir(character_path):
                    self.train_lines.append(os.path.join(character_path, image))
                    self.train_labels.append(self.types)
                self.types += 1
        else:
            #-------------------------------------------------------------#
            #   Omniglot数据集，遍历大循环
            #-------------------------------------------------------------#
            for alphabet in os.listdir(train_path):
                alphabet_path = os.path.join(train_path, alphabet)
                #-------------------------------------------------------------#
                #   Omniglot数据集，遍历小循环
                #-------------------------------------------------------------#
                for character in os.listdir(alphabet_path):
                    character_path = os.path.join(alphabet_path, character)
                    #-------------------------------------------------------------#
                    #   对每张图片进行遍历
                    #-------------------------------------------------------------#
                    for image in os.listdir(character_path):
                        self.train_lines.append(os.path.join(character_path, image))
                        self.train_labels.append(self.types)
                    self.types += 1

        # #-------------------------------------------------------------#
        # #   将获得的所有图像进行打乱。
        # #-------------------------------------------------------------#
        # random.seed(1)
        # shuffle_index = np.arange(len(self.train_lines), dtype=np.int32)
        # shuffle(shuffle_index)
        # random.seed(None)
        # self.train_lines    = np.array(self.train_lines,dtype=np.object)
        # self.train_labels   = np.array(self.train_labels)
        # self.train_lines    = self.train_lines[shuffle_index]
        # self.train_labels   = self.train_labels[shuffle_index]
        
        # #-------------------------------------------------------------#
        # #   将训练集和验证集进行划分
        # #-------------------------------------------------------------#
        # self.val_lines      = self.train_lines[self.num_train:]
        # self.val_labels     = self.train_labels[self.num_train:]
    
        # self.train_lines    = self.train_lines[:self.num_train]
        # self.train_labels   = self.train_labels[:self.num_train]
        #-------------------------------------------------------------#
        #   将训练集和验证集进行划分
        #-------------------------------------------------------------#
        self.val_lines      = self.train_lines[self.num_train:]
        self.val_labels     = self.train_labels[self.num_train:]
    
        self.train_lines    = self.train_lines[:self.num_train]
        self.train_labels   = self.train_labels[:self.num_train]
        #-------------------------------------------------------------#
        #   将获得的所有图像进行打乱。
        #-------------------------------------------------------------#
        random.seed(1)
        shuffle_index = np.arange(len(self.train_lines), dtype=np.int32)
        shuffle(shuffle_index)
        random.seed(None)
        self.train_lines    = np.array(self.train_lines,dtype=np.object)
        self.train_labels   = np.array(self.train_labels)
        self.train_lines    = self.train_lines[shuffle_index]
        self.train_labels   = self.train_labels[shuffle_index]
        
        shuffle_index_val = np.arange(len(self.val_lines), dtype=np.int32)
        shuffle(shuffle_index_val)
        random.seed(None)
        self.val_lines    = np.array(self.val_lines,dtype=np.object)
        self.val_labels   = np.array(self.val_labels)
        self.val_lines    = self.val_lines[shuffle_index_val]
        self.val_labels   = self.val_labels[shuffle_index_val]
    def __getitem__(self, index):
        
        if self.train:
            lines = self.train_lines
            labels = self.train_labels
        else:
            lines = self.val_lines
            labels = self.val_labels
    
        batch_images_path = []
        #------------------------------------------#
        #   首先选取两张类别相同的图片
        #------------------------------------------#
        c               = random.randint(0, self.types - 1)
        selected_path   = lines[labels[:] == c]                 #在train的数据中心 label为c的数据所在的path
        while len(selected_path)<2:
            c               = random.randint(0, self.types - 1)
            selected_path   = lines[labels[:] == c]

        image_indexes = random.sample(range(0, len(selected_path)), 2)      #从所有的label为c的数据中心取出2张图片
        #------------------------------------------#
        #   取出两张类似的图片
        #   对于这两张图片，网络应当输出1
        #------------------------------------------#
        batch_images_path.append(selected_path[image_indexes[0]])
        batch_images_path.append(selected_path[image_indexes[1]])
#--------------------------------------------------------------------------------------------
        if True:
            
            different_c         = list(range(self.types))
            different_c.pop(c)
            different_c_index   = np.random.choice(range(0, self.types - 1), 1)                 #选一个除了c之外的其他类
            current_c           = different_c[different_c_index[0]]
            selected_path       = lines[labels == current_c]
            while len(selected_path)<1:
                different_c_index   = np.random.choice(range(0, self.types - 1), 1)
                current_c           = different_c[different_c_index[0]]
                selected_path       = lines[labels == current_c]

            image_indexes = random.sample(range(0, len(selected_path)), 1)
            batch_images_path.append(selected_path[image_indexes[0]])
#--------------------------------------------------------------------------------------------
        if False:
            #线下生成的难例样本
            image3_files_path = './datasets/expend/expend_lines_35000/hard_lines' + batch_images_path[0].split('lines')[-1][:-7]

            image3_file = random.choice(os.listdir(image3_files_path))
            batch_images_path.append(os.path.join(image3_files_path,image3_file))
#----------------------------------------------------------------------------------------------------------


        if False:
            #通过皮尔逊相关系数选择难例样本




#-----------------------------------------------------------------------------------------------------------
        # if batch_images_path[1] == './datasets/expend/expend_lines_5000/lines\\00004132\\002.jpg':
        #     print('1')
        # if batch_images_path[0] == './datasets/expend/expend_lines_5000/lines\\00004132\\002.jpg':
        #     print('1')
        #img1 anchor
        img1 = Image.open(batch_images_path[0])
        reimg1 = img1.resize((self.image_width,self.image_height))
        # image1 = np.asarray(reimg1).astype(np.float64)
        # image1 = np.transpose(image1, [2, 0, 1])
        # image1 = image1 / 255

        image1 =  self.transform(reimg1 )

        img2 = Image.open(batch_images_path[1])
        reimg2 = img2.resize((self.image_width,self.image_height))
        image2 =  self.transform(reimg2)

        img3 = Image.open(batch_images_path[2])
        reimg3 = img3.resize((self.image_width,self.image_height))
        image3 =  self.transform(reimg3)

        return (image1, image2, image3), []
        

# DataLoader中collate_fn使用
def triplet_dataset_collate(batch):
    left_images     = []
    right_images    = []
    labels          = []
    for pair_imgs, pair_labels in batch:
        for i in range(len(pair_imgs[0])):
            left_images.append(pair_imgs[0][i])
            right_images.append(pair_imgs[1][i])
            labels.append(pair_labels[i])
            
    return np.array([left_images, right_images]), np.array(labels)
