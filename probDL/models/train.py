import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import cv2
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import os
from torch import optim

class TrainModel():
    def __init__(self,model_name=None,device = 'gpu',epochs = 100,loss = nn.CrossEntropyLoss(),optimizer=None):
        self.device = torch.device('cuda' if device=='gpu' and torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.loss = loss
        self.model_name = model_name
        self.optimizer = optimizer
        self.params={'epoch':0,'loss':0}
    def train(self,train_ds,bs):
        
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = (self.model_name).to(self.device)
        opt = self.optimizer
        loss = self.loss
        train_loader = data.DataLoader(train_ds, batch_size=bs,sampler=RandomSampler(train_ds))
        self.params['train loss'] = []
        self.params['Epoch'] = []
        for epoch in range(self.epochs):
            train_loss = 0
            for x,y in train_loader:
                opt.zero_grad()
                x = x.reshape((bs,-1,train_ds.image_size[0],train_ds.image_size[1]))
                x = x.to(self.device)
                y = y.to(self.device)
                score,var = model(x)
                loss_val = loss(score,y,var)
                loss_val.backward()
                opt.step()
                train_loss = train_loss + loss_val.item()
            self.params['train loss'].append(train_loss)
            self.params['Epoch'].append(epoch)
            print('Epoch: ',epoch,' ','Train Loss: ',train_loss/len(train_loader))
            self.mode_name = model
        
    
    def test(self,test_ds,bs):
        test_loader = data.DataLoader(test_ds, batch_size=bs,sampler=RandomSampler(test_ds))
        test_loss = 0
        for x,y in test_loader:
                x = x.reshape((bs,-1,train_ds.test_size[0],test_ds.image_size[1]))
                x = x.to(self.device)
                y = y.to(self.device)
                score,var = model(x)
                loss_val = loss(score,y,var)
                test_loss = test_loss + loss_val.item()
        print('Test Loss: ',test_loss/len(test_loader))
    
    def predict_segmentation_mask(self,input_image_location,data_load_segment=None):
        image_size = data_load_segment.image_size
        img = cv2.imread(input_image_location)
        resize_img = cv2.resize(img, (image_size[0],image_size[1]), cv2.INTER_NEAREST) 
        processed_img = torch.from_numpy(resize_img).float()
        processed_img = processed_img.reshape((1,-1,image_size[0],image_size[1]))
        predict_mask,uncertain = self.model_name(processed_img.to(self.device))
        return predict_mask
    
    def predict_aleatoric(self,input_image_location,data_load_segment=None):
        image_size = data_load_segment.image_size
        img = cv2.imread(input_image_location)
        resize_img = cv2.resize(img, (image_size[0],image_size[1]), cv2.INTER_NEAREST) 
        processed_img = torch.from_numpy(resize_img).float()
        processed_img = processed_img.reshape((1,-1,image_size[0],image_size[1]))
        predict_mask,aleatoric = self.model_name(processed_img.to(self.device))
        aleatoric = F.softmax(aleatoric,dim=1)
        return aleatoric
    
    def predict_epistemic(self,input_image_location,num_samples,data_load_segment=None):
        image_size = data_load_segment.image_size
        num_class = data_load_segment.num_class
        img = cv2.imread(input_image_location)
        resize_img = cv2.resize(img, (image_size[0],image_size[1]), cv2.INTER_NEAREST) 
        processed_img = torch.from_numpy(resize_img).float()
        processed_img = processed_img.reshape((1,-1,image_size[0],image_size[1]))
        
        num_predict = torch.zeros(num_samples,num_class,image_size[0],image_size[1])
        for i in range(num_samples):
            predict_mask,aleatoric = self.model_name(processed_img.to(self.device))
            num_predict[i] = F.softmax(predict_mask,dim=1)
        epistemic = num_predict.var(0)
        epistemic = F.softmax(epistemic,dim=1)
        return epistemic
