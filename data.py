import torch.utils.data as data
import os
import nibabel as nib
import skimage.transform as trans


class SpleenDataset(data.Dataset):

    def __init__(self,transform = None,target_transform = None):
        filename_list = os.listdir('dataset/spleen/imagesTr')  # data path
        n = len(filename_list)  # num of volumes
        
        imgs = []
        for i in range(n):
            filename_i = filename_list[i]
            img = os.path.join('dataset/spleen/imagesTr/%s' % filename_i)   # image path
            mask = os.path.join('dataset/spleen/labelsTr/%s' % filename_i)  # label path
            imgs.append([img,mask])
        
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self,index):
        x_path,y_path = self.imgs[index]
        img_x = nib.load((x_path)).get_fdata() # resize images to same size
        img_y = nib.load((y_path)).get_fdata()
        img_x = trans.resize(img_x,(512,512,96))
        img_y = trans.resize(img_y,(512,512,96))
        # sagittal
        
        # axial
        #img_x = np.transpose(img_x,(2,0,1))
        #img_y = np.transpose(img_y,(2,0,1))
        
        # coronal
        #img_x = np.transpose(img_x,(1,0,2))
        #img_y = np.transpose(img_y,(1,0,2))

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x,img_y
    
    
    def __len__(self):
        return len(self.imgs)


class SpleenDatasetTest(data.Dataset):

    def __init__(self,transform = None,target_transform = None):
        filename_list = os.listdir('dataset/spleen/imagesTs')  # data path
        n = len(filename_list)  # num of volumes
        
        imgs = []
        for i in range(n):
            filename_i = filename_list[i]
            img = os.path.join('dataset/spleen/imagesTs/%s' % filename_i)   # image path
            mask = os.path.join('dataset/spleen/labelsTs/%s' % filename_i)  # label path
            imgs.append([img,mask])
        
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self,index):
        x_path,y_path = self.imgs[index]
        img_x = nib.load((x_path)).get_fdata() # resize images to same size
        img_y = nib.load((y_path)).get_fdata()
        img_x = trans.resize(img_x,(512,512,96))
        img_y = trans.resize(img_y,(512,512,96))
        # sagittal
        
        # axial
        #img_x = np.transpose(img_x,(2,0,1))
        #img_y = np.transpose(img_y,(2,0,1))
        
        # coronal
        #img_x = np.transpose(img_x,(1,0,2))
        #img_y = np.transpose(img_y,(1,0,2))

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x,img_y
    
    
    def __len__(self):
        return len(self.imgs)
