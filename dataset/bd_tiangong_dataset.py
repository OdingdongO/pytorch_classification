# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import PIL.Image as Image
from data_aug import *
import cv2
class2label={"DESERT":0,"MOUNTAIN":1,"OCEAN":2,"FARMLAND":3,"LAKE":4,"CITY":5}
label2class=["DESERT","MOUNTAIN","OCEAN","FARMLAND","LAKE","CITY"]

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

        img =cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   # [h,w,3]  RGB

        if self.transforms is not None:
            img = self.transforms(img)
        label = self.labels[item]
        # plt.imshow(img)
        # plt.show()
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
    import sys
    from sklearn.model_selection import train_test_split
    reload(sys)
    sys.setdefaultencoding('utf8')
    label_csv ="/media/hszc/model/detao/data/dianshi/宽波段数据集-预赛训练集2000/光学数据集-预赛训练集-2000-有标签.csv"
    imgroot="/media/hszc/model/detao/data/dianshi/宽波段数据集-预赛训练集2000/预赛训练集-2000"
    label = pd.read_csv(label_csv,names=["ImageName","class"])
    label["label"]=label["class"].apply(lambda x:class2label[x])
    print(label["label"].value_counts())
    train_pd,val_pd = train_test_split(label, test_size=0.15, random_state=43,stratify=label["label"])
    data_set = {}
    data_set['val'] = dataset(imgroot,anno_pd=train_pd,transforms=None)
    for data in data_set["val"]:
        image, label =data
        print(label)
