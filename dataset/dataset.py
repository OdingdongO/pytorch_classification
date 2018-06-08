# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import PIL.Image as Image
from data_aug import *
import cv2
class dataset(data.Dataset):
    def __init__(self, imgroot, anno_pd, transforms=None):
        self.root_path = imgroot
        self.paths = anno_pd['ImageName'].tolist()
        self.labels = anno_pd['label'].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        # img = self.pil_loader(img_path)
        img =cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   # [h,w,3]  RGB

        if self.transforms is not None:
            img = self.transforms(img)
        label = self.labels[item]-1
        # return img, label
        return torch.from_numpy(img).float(), label

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
def collate_fn(batch):
    imgs = []
    label = []

    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    rawdata_root = '/media/hszc/data/detao/data/baidu/datasets'
    train_pd = pd.read_csv("/media/hszc/data/detao/data/baidu/datasets/train.txt", sep=" ",
                           header=None, names=['ImageName', 'label'])
    val_pd = pd.read_csv("/media/hszc/data/detao/data/baidu/datasets/test_answer.txt", sep=" ",
                         header=None, names=['ImageName', "label"])
    val_pd['ImageName'] = val_pd['ImageName'].apply(lambda x: os.path.join('test', x))
    train_pd['ImageName'] = train_pd['ImageName'].apply(lambda x: os.path.join('train', x))
    data_transforms = {
        'train': Compose([
            RandomRotate(angles=(-15,15)),
            ExpandBorder(size=(368, 368), resize=True),
            RandomResizedCrop(size=(336, 336)),
            # RandomHflip(),
            # transforms.RandomResizedCrop(336,scale=(0.49,1.0)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),   # 0-255 to 0-1
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': Compose([
            ExpandBorder(size=(368, 368), resize=True),
            # CenterCrop(size=(336, 336)),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_set = {}
    data_set['train'] = dataset(imgroot=rawdata_root, anno_pd=train_pd,
                                transforms=data_transforms["train"],
                                )
    data_set['val'] = dataset(imgroot=rawdata_root, anno_pd=val_pd,
                              transforms=data_transforms["val"],
                              )
    for data in data_set["val"]:
        print("22222")