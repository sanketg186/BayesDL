import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import cv2
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import os
from torch import optim
from torch.distributions import Normal,MultivariateNormal

class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path,num_class,image_size=(256,256)):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = os.listdir(os.path.join(folder_path,'train'))
        self.mask_files =  os.listdir(os.path.join(folder_path,'train_mask'))
        self.num_class = num_class
        self.folder_path = folder_path
        self.image_size = image_size

    def __getitem__(self, index):
        img_path = os.path.join(self.folder_path,'train/',self.img_files[index])
        mask_path = os.path.join(self.folder_path,'train_mask/',self.mask_files[index])
        img = cv2.imread(img_path)
        resize_img = cv2.resize(img, (self.image_size[0], self.image_size[1]), cv2.INTER_NEAREST)
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        one_hot =self.one_hot_encode(mask,self.num_class)
        return torch.from_numpy(resize_img).float(), torch.from_numpy(mask).long()
    
    def one_hot_encode(self,mask,num_class):
        one_hot = torch.zeros(num_class,mask.shape[0],mask.shape[1])
        for i in range(num_class):
            one_hot[i,mask==i] = 1
        return one_hot

    def __len__(self):
        return len(self.img_files)