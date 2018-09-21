#coding=utf-8
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.bd_xjtu_dataset import collate_fn, dataset
import torch
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train_util import train, trainlog
from  torch.nn import CrossEntropyLoss
import logging
from dataset.data_aug import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
http://dianshi.baidu.com/gemstone/competitions/detail?raceId=17
'''
save_dir = '/media/hszc/model/detao/baidu_model/resnet50'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = '%s/trainlog.log'%save_dir
trainlog(logfile)

rawdata_root = '/media/hszc/data/detao/data/baidu/datasets/train'
all_pd = pd.read_csv("/media/hszc/data/detao/data/baidu/datasets/train.txt",sep=" ",
                     header=None, names=['ImageName', 'label'])
train_pd, val_pd = train_test_split(all_pd, test_size=0.15, random_state=43,
                                    stratify=all_pd['label'])
print(val_pd.shape)

'''数据扩增'''
data_transforms = {
    'train': Compose([
        RandomRotate(angles=(-15,15)),
        ExpandBorder(size=(368,368),resize=True),
        RandomResizedCrop(size=(336, 336)),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': Compose([
        ExpandBorder(size=(336,336),resize=True),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_set = {}
data_set['train'] = dataset(imgroot=rawdata_root,anno_pd=train_pd,
                           transforms=data_transforms["train"],
                           )
data_set['val'] = dataset(imgroot=rawdata_root,anno_pd=val_pd,
                           transforms=data_transforms["val"],
                           )
dataloader = {}
dataloader['train']=torch.utils.data.DataLoader(data_set['train'], batch_size=4,
                                               shuffle=True, num_workers=4,collate_fn=collate_fn)
dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=4,
                                               shuffle=True, num_workers=4,collate_fn=collate_fn)
'''model'''
model =resnet50(pretrained=True)
model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
model.fc = torch.nn.Linear(model.fc.in_features,100)

base_lr =0.001
resume =None
if resume:
    logging.info('resuming finetune from %s'%resume)
    model.load_state_dict(torch.load(resume))
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-5)
criterion = CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

train(model,
      epoch_num=50,
      start_epoch=0,
      optimizer=optimizer,
      criterion=criterion,
      exp_lr_scheduler=exp_lr_scheduler,
      data_set=data_set,
      data_loader=dataloader,
      save_dir=save_dir,
      print_inter=50,
      val_inter=400)