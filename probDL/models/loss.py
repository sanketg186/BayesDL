import torch.nn as nn
from torch.distributions import Normal,MultivariateNormal
import torch

class Bayes_loss(nn.Module):
    def __init__(self,num_samples=10,num_classes=2,weight=None,output_shape=(256,256)):
        super(Bayes_loss,self).__init__()
        self.num_samples=num_samples
        self.num_classes = num_classes
        self.weight = weight
        self.cross_entropy = nn.CrossEntropyLoss(self.weight)
        self.elu=nn.ELU()
        self.output_shape = output_shape
    
    def gaussian_cross_entropy(self,dist,outputs,targets,undistorted_loss,std):
        std_samples = dist.sample_n(self.num_classes)
        std_samples = std_samples.view(1,self.num_classes,self.output_shape[0],self.output_shape[1])
        std_samples = std_samples.to('cuda:0')
        std_samples =std*std_samples
        temp = outputs + std_samples #.unsqueeze(0)
        distorted_loss = self.cross_entropy(temp, targets)
        diff = undistorted_loss - distorted_loss
        return -self.elu(diff)

     
    def forward(self,outputs,targets,log_var):
        variance=log_var
        std = torch.sqrt(variance+.0000001) #size 32 x 128 x 128
        variance_depressor = torch.exp(variance) - torch.ones_like(variance)
        variance_depressor = variance_depressor.view(1,-1)
        variance_depressor=torch.mean(variance_depressor) # scalar value
        
        undistorted_loss = self.cross_entropy(outputs,targets)

        mean=torch.zeros(self.output_shape[0],self.output_shape[1]).to('cuda:0')
        std1=torch.eye(self.output_shape[0],self.output_shape[1]).to('cuda:0')
        dist = MultivariateNormal(mean,std1)
        #dist = Normal(torch.zeros_like(std),std)
        #samples = dist.sample(12)
        mc_sample=0
        for i in range(self.num_samples):
            mc_sample =mc_sample+self.gaussian_cross_entropy(dist,outputs,targets,undistorted_loss,std)

        variance_loss = ((mc_sample/self.num_samples) * undistorted_loss)
    
        return variance_loss+ undistorted_loss +variance_depressor
