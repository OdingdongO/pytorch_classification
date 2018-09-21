# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import PIL.Image as Image
from data_aug import *
import cv2
import numpy as np
import random
import glob
import pandas as pd
defect_label_order = ['norm', 'defect1', 'defect2', 'defect3', 'defect4', 'defect5', 'defect6', 'defect7',
                      'defect8', 'defect9', 'defect10', 'defect11']
defect_code = {
    '正常':    'norm',
    '不导电':  'defect1',
    '擦花':    'defect2',
    '横条压凹': 'defect3',
    '桔皮': 'defect4',
    '漏底':    'defect5',
    '碰伤':   'defect6',
    '起坑':   'defect7',
    '凸粉': 'defect8',
    '涂层开裂': 'defect9',
    '脏点': 'defect10',
    '其他':   'defect11'
}
defect_label = {
    '正常':    '0',
    '不导电':  '1',
    '擦花':    '2',
    '横条压凹': '3',
    '桔皮': '4',
    '漏底': '5',
    '碰伤': '6',
    '起坑': '7',
    '凸粉': '8',
    '涂层开裂':'9',
    '脏点': '10',
    '其他': '11'
}
label2defect_map = dict(zip(defect_label.values(), defect_label.keys()))

def get_image_pd(img_root):
    img_list = glob.glob(img_root + "/*/*.jpg")
    img_list2 = glob.glob(img_root+ "/*/*/*.jpg")

    image_pd1 = pd.DataFrame(img_list, columns=["ImageName"])

    image_pd2 = pd.DataFrame(img_list2, columns=["ImageName"])
    image_pd1["label_name"]=image_pd1["ImageName"].apply(lambda x:x.split("/")[-2])
    image_pd2["label_name"]=image_pd2["ImageName"].apply(lambda x:x.split("/")[-3])
    all_pd=image_pd1.append(image_pd2)
    all_pd["label"]=all_pd["label_name"].apply(lambda x:defect_label[x])
    print(all_pd["label"].value_counts())
    return all_pd

class dataset(data.Dataset):
    def __init__(self, anno_pd, transforms=None,debug=False,test=False):
        self.paths = anno_pd['ImageName'].tolist()
        self.labels = anno_pd['label'].tolist()
        self.transforms = transforms
        self.debug=debug
        self.test=test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path =self.paths[item]
        img_id =img_path.split("/")[-1]
        img =cv2.imread(img_path) #BGR

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   # [h,w,3]  RGB

        if self.transforms is not None:
            img = self.transforms(img)
        label = self.labels[item]
        if self.debug:
            print(label)
            plt.imshow(img)
            plt.show()
        if self.test:
            return torch.from_numpy(img).float(), int(label)
        else:
            return torch.from_numpy(img).float(), int(label)


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
    rawdata_root = '/media/hszc/model/detao/data/guangdong/guangdong_round1_train2_2018091622/瑕疵样本'
    all_pd= get_image_pd(rawdata_root)
    train_pd,val_pd = train_test_split(all_pd, test_size=0.15, random_state=43,stratify=all_pd["label"])

    print(val_pd["label"].value_counts())

    class train_Aug(object):
        def __init__(self):
            self.augment = Compose([
                Resize(size=(640, 640)),
                FixRandomRotate(bound='Random'),
                RandomHflip(),
                RandomVflip(),
                # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        def __call__(self, *args):
            return self.augment(*args)
    data_set = {}
    data_set['val'] = dataset(anno_pd=train_pd,transforms=train_Aug(),debug=True)
    for data in data_set["val"]:
        image, label =data
        print(label)