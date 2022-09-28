import os
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import transforms
import numpy as np
import torch
import random
from gridmask import GridMask


class ObjDataset(data.Dataset):
    def __init__(self, images, gts, trainsize, mode):
        self.trainsize = trainsize
        self.images = images
        self.mode = mode
        self.gts = gts
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.gridmask = GridMask()
        self.img_transform_w = transforms.Compose([
            transforms.Resize((self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform_w = transforms.Compose([
            transforms.Resize((self.trainsize)),
            transforms.ToTensor()])
        
        
        
        self.img_transform_s = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees = 90,translate=(0.5,0.5),shear=30),
            transforms.ColorJitter(hue = 0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(3),
            transforms.Resize((self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform_s = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees = 90,translate=(0.5,0.5),shear=30),
            transforms.Resize((self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed =  np.random.randint(2147483647)
        

        
        if self.mode == 'weak':
            
            image = self.img_transform_w(image)
            gt = self.gt_transform_w(gt)

        if self.mode =='strong':
            torch.manual_seed(seed)
            image = self.img_transform_s(image)
            torch.manual_seed(seed)
            gt = self.gt_transform_s(gt)
            image = self.gridmask(image)
            

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class ValObjDataset(data.Dataset):
    def __init__(self, images, gts, trainsize):
        self.trainsize = trainsize
        self.images = images
        self.gts = gts
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize)),
            transforms.ToTensor()])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < 256 or w < 256:
            h = max(h, 256)
            w = max(w, 256)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def image_loader(image_root, gt_root,val_img_root,val_gt_root, batch_size, image_size, split=1, labeled_ratio=0.05,mode='weak_1'):
    images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
    gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
    
    val_img = [val_img_root+ f for f in os.listdir(val_img_root) if f.endswith('.jpg') or f.endswith('.png')]
    val_label = [val_gt_root+ f for f in os.listdir(val_gt_root) if f.endswith('.jpg') or f.endswith('.png')]
    
    train_images = images[0:int(len(images) * split)]
    val_images = val_img[0:int(len(val_img) * split)] 
    val_gts = val_label[0:int(len(val_label) * split)]
    train_gts = gts[0:int(len(images) * split)]


    labeled_train_images = train_images[0:int(len(train_images) * labeled_ratio)] 
    labeled_train_images_1 = labeled_train_images[0:int(len(labeled_train_images) * 0.5)] 
    labeled_train_images_2 = labeled_train_images[int(len(labeled_train_images) * 0.5):]
    unlabeled_train_images = train_images[int(len(train_images) * labeled_ratio):] 
    labeled_train_gts = train_gts[0:int(len(train_gts) * labeled_ratio)]
    labeled_train_gts_1 = labeled_train_gts[0:int(len(labeled_train_gts) * 0.5)]
    labeled_train_gts_2 = labeled_train_gts[int(len(labeled_train_gts) * 0.5):]
    unlabeled_train_gts = train_gts[int(len(train_gts) * labeled_ratio):]

    labeled_train_dataset_1 = ObjDataset(labeled_train_images_1, labeled_train_gts_1, image_size,mode='weak')
    labeled_train_dataset_2 = ObjDataset(labeled_train_images_2, labeled_train_gts_2, image_size,mode='weak')
    unlabeled_train_dataset = ObjDataset(unlabeled_train_images, unlabeled_train_gts, image_size,mode='strong')
    val_dataset = ValObjDataset(val_images, val_gts, image_size)

    labeled_data_loader_1 = data.DataLoader(dataset=labeled_train_dataset_1,
                                  batch_size=batch_size,
                                  num_workers=1,
                                  pin_memory=True,
                                  shuffle=True)

    labeled_data_loader_2 = data.DataLoader(dataset=labeled_train_dataset_2,
                                            batch_size=batch_size,
                                            num_workers=1,
                                            pin_memory=True,
                                            shuffle=True)

    unlabeled_data_loader = data.DataLoader(dataset=unlabeled_train_dataset,
                                          batch_size=batch_size,
                                          num_workers=1,
                                          pin_memory=True,
                                          shuffle=True)

    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 num_workers=1,
                                 pin_memory=True,
                                 shuffle=False)

    return labeled_data_loader_1, labeled_data_loader_2, unlabeled_data_loader, val_loader