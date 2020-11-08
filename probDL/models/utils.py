import pandas as pd
class Utils:
    def convert_to_label(self,mask,df):
        label = np.zeros((256,256,3))
        for i in range(256):
            for j in range(256):
                cl = mask[i][j]
                r = int(df[df['class']==cl]['r'])
                g = int(df[df['class']==cl]['g'])
                b = int(df[df['class']==cl]['b'])
                label[i,j,0] = r
                label[i,j,1] = g
                label[i,j,2] = b
        return label/255
    
    def convert_to_mask(self,label,df):
        mask = np.zeros((256,256))
        for i in range(256):
            for j in range(256):
                for name in df['name']:
                    r = int(df[df['name']==name]['r'])
                    g = int(df[df['name']==name]['g'])
                    b = int(df[df['name']==name]['b'])
                    class_label = int(df[df['name']==name]['class'])
                    if label[i,j,0]==b and label[i,j,1]==g and label[i,j,2]==r:
                        mask[i,j]= class_label
        return mask
    
    def onehot_to_label(self,mask,num_class):
        label = np.zeros((256,256))
        for i in range(num_class):
            label[mask[i]==1]=i
        return label