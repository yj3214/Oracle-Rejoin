import os
import random
from random import shuffle
from PIL import Image
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class SiameseDataset(Dataset):
    def __init__(self, input_shape, dataset_path, num_train, num_val, train=True, train_own_data=True):
        super(SiameseDataset, self).__init__()

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
        # self.num_val        = num_val
        self.train          = train
        self.train_own_data = train_own_data
        self.load_dataset()

    def __len__(self):
        return self.num_train
        # if self.train:
        #     return self.num_train
        # else:
        #     return self.num_val

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
        
        #-------------------------------------------------------------#
        #   将训练集和验证集进行划分
        #-------------------------------------------------------------#
        self.val_lines      = self.train_lines[self.num_train:]
        self.val_labels     = self.train_labels[self.num_train:]
    
        self.train_lines    = self.train_lines[:self.num_train]
        self.train_labels   = self.train_labels[:self.num_train]


    def _convert_path_list_to_images_and_labels(self, path_list):
        #-------------------------------------------#
        #   如果batch_size = 16
        #   len(path_list)/2 = 32
        #-------------------------------------------#
        number_of_pairs = int(len(path_list) / 2)
        #-------------------------------------------#
        #   定义网络的输入图片和标签
        #-------------------------------------------#
        pairs_of_images = [np.zeros((number_of_pairs, self.channel, self.image_height, self.image_width)) for i in range(2)]
        labels          = np.zeros((number_of_pairs, 1))

        #-------------------------------------------#
        #   对图片对进行循环
        #   0,1为同一种类，2,3为不同种类
        #   4,5为同一种类，6,7为不同种类
        #   以此类推
        #-------------------------------------------#
        for pair in range(number_of_pairs):
            #-------------------------------------------#
            #   将图片填充到输入1中
            #-------------------------------------------#
            #image = Image.open(path_list[pair * 2])
            #image.show()
            img = cv2.imread(path_list[pair * 2])

            height, width,_ = img.shape

            if height > width:
                reimg = cv2.copyMakeBorder(img,0,0,(height - width)//2,(height - width)//2,cv2.BORDER_CONSTANT, value=[255,255,255])
                reimg = cv2.resize(reimg,(self.image_width,self.image_height))

            else:
                reimg = cv2.copyMakeBorder(img,(width - height)//2,(width - height)//2,0,0,cv2.BORDER_CONSTANT, value=[255,255,255])
                reimg = cv2.resize(reimg,(self.image_width,self.image_height))

            img_nummpy = np.array(reimg)
            image = Image.fromarray(img_nummpy.astype('uint8')).convert('RGB')
            # print(image.size)
            # image.save('123131.jpg')
            # image.show()

            # image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64)
            image = np.transpose(image, [2, 0, 1])
            image = image / 255
            if self.channel == 1:
                pairs_of_images[0][pair, 0, :, :] = image
            else:
                pairs_of_images[0][pair, :, :, :] = image

            #-------------------------------------------#
            #   将图片填充到输入2中
            #-------------------------------------------#
            # image = Image.open(path_list[pair * 2 + 1])

            img = cv2.imread(path_list[pair * 2 + 1])

            height, width,_ = img.shape

            if height > width:
                reimg = cv2.copyMakeBorder(img,0,0,(height - width)//2,(height - width)//2,cv2.BORDER_CONSTANT, value=[255,255,255])
                reimg = cv2.resize(reimg,(self.image_width,self.image_height))

            else:
                reimg = cv2.copyMakeBorder(img,(width - height)//2,(width - height)//2,0,0,cv2.BORDER_CONSTANT, value=[255,255,255])
                reimg = cv2.resize(reimg,(self.image_width,self.image_height))

            img_nummpy = np.array(reimg)
            image = Image.fromarray(img_nummpy.astype('uint8')).convert('RGB')
            


            # image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64)
            image = np.transpose(image, [2, 0, 1])
            image = image / 255
            if self.channel == 1:
                pairs_of_images[1][pair, 0, :, :] = image
            else:
                pairs_of_images[1][pair, :, :, :] = image
                
            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1
            # print(labels[pair])

        #-------------------------------------------#
        #   随机的排列组合
        #-------------------------------------------#
        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_permutation, :, :, :]
        pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_permutation, :, :, :]
        return pairs_of_images, labels

    def __getitem__(self, index):

        lines = self.train_lines
        labels = self.train_labels        
        # if self.train:
        #     lines = self.train_lines
        #     labels = self.train_labels
        # else:
        #     lines = self.val_lines
        #     labels = self.val_labels
    
        batch_images_path = []
        #------------------------------------------#
        #   首先选取三张类别相同的图片
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

        #------------------------------------------#
        #   取出两张不类似的图片
        #------------------------------------------#
        # batch_images_path.append(selected_path[image_indexes[1]])
        #------------------------------------------#
        #   取出与当前的小类别不同的类
        #------------------------------------------#
        # different_c         = list(range(self.types))
        # different_c.pop(c)
        # different_c_index   = np.random.choice(range(0, self.types - 1), 1)                 #选一个除了c之外的其他类
        # current_c           = different_c[different_c_index[0]]
        # selected_path       = lines[labels == current_c]
        # while len(selected_path)<1:
        #     different_c_index   = np.random.choice(range(0, self.types - 1), 1)
        #     current_c           = different_c[different_c_index[0]]
        #     selected_path       = lines[labels == current_c]

        # image_indexes = random.sample(range(0, len(selected_path)), 1)
        # batch_images_path.append(selected_path[image_indexes[0]])
        
        images, labels = self._convert_path_list_to_images_and_labels(batch_images_path)
        return images, labels
        

# DataLoader中collate_fn使用
def dataset_collate(batch):
    left_images     = []
    right_images    = []
    labels          = []
    for pair_imgs, pair_labels in batch:
        for i in range(len(pair_imgs[0])):
            left_images.append(pair_imgs[0][i])
            right_images.append(pair_imgs[1][i])
            labels.append(pair_labels[i])
            
    return np.array([left_images, right_images]), np.array(labels)
