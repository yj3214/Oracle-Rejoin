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

class TripletDataset_online(Dataset):
    def __init__(self, input_shape, dataset_path, num_train, num_val, train=True, train_own_data=True):
        super(TripletDataset_online, self).__init__()

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
        # print('1')
        


    def __getitem__(self, index):
        
        if self.train:
            line = self.train_lines [index]
            label = self.train_labels[index]
            # lines = self.train_lines
            # labels = self.train_labels
        else:
            line = self.val_lines [index]
            label = self.val_labels[index]
            # lines = self.val_lines
            # labels = self.val_labels
        
        #img1 anchor
        img1 = Image.open(line)
        reimg1 = img1.resize((self.image_width,self.image_height))

        image1 =  self.transform(reimg1 )

        return image1, label
        

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
