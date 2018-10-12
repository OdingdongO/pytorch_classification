#coding=utf-8
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.pdr_dataset import collate_fn, dataset
import torch
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50,resnet101
from models.xception import pdr_xception
from models.inception_v4 import inceptionv4
from models.multiscale_resnet import multiscale_resnet
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train_util import train, trainlog
from  torch.nn import CrossEntropyLoss
from  utils.losses import FocalLoss
import logging
from dataset.data_aug import *
import sys
import argparse
reload(sys)
sys.setdefaultencoding('utf8')
'''
https://challenger.ai/competition/pdr2018
'''
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')

parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--checkpoint_dir', type=str, default='/media/hszc/model/detao/models/pdr/resnet101_32', help='directory where model checkpoints are saved')
parser.add_argument('--cuda_device', type=str, default="0,3", help='whether to use cuda if available')
parser.add_argument('--net', dest='net',type=str, default='resnet101',help='resnet101,resnet50,xception,inceptionv4')
parser.add_argument('--loss', dest='loss',type=str, default='CrossEntropyLoss',help='focal_loss,CrossEntropyLoss')
parser.add_argument('--optim', dest='optim',type=str, default='sgd',help='adam,sgd')
parser.add_argument('--resume', type=str, default="/media/hszc/model/detao/models/pdr/resnet101/weights-6-1718-[0.8310].pth", help='path to resume weights file')
# parser.add_argument('--resume', type=str, default="/media/hszc/model/detao/models/pdr/inceptionv4/weights-9-1577-[0.8312].pth", help='path to resume weights file')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')

parser.add_argument('--save_checkpoint_val_interval', type=int, default=500, help='interval between saving model weights')
parser.add_argument('--print_interval', type=int, default=100, help='interval between print log')
parser.add_argument('--img_root_train', type=str, default= "/media/hszc/model/detao/data/pdr2/ai_challenger_pdr2018_trainingset_20180905/AgriculturalDisease_trainingset", help='whether to img root')
parser.add_argument('--img_root_val', type=str, default= "/media/hszc/model/detao/data/pdr2/ai_challenger_pdr2018_validationset_20180905/AgriculturalDisease_validationset", help='whether to img root')

parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device
if __name__ == '__main__':

    # # saving dir
    save_dir = opt.checkpoint_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logfile = '%s/trainlog.log' % save_dir
    trainlog(logfile)
    logging.info(opt)

    train_info = open(os.path.join(opt.img_root_train, "AgriculturalDisease_train_annotations.json"))
    val_info = open(os.path.join(opt.img_root_val, "AgriculturalDisease_validation_annotations.json"))

    train_pd = pd.read_json(train_info)
    val_pd = pd.read_json(val_info)
    train_pd["label"] = train_pd['disease_class']
    val_pd["label"] = val_pd['disease_class']

    train_pd["ImageName"]=train_pd["image_id"].apply(lambda x:os.path.join(opt.img_root_train,"images",x))
    val_pd["ImageName"]=val_pd["image_id"].apply(lambda x:os.path.join(opt.img_root_val,"images",x))

    print(val_pd.shape)

    '''数据扩增'''
    if opt.net == "resnet50" :

        data_transforms = {
            'train': Compose([
                RandomRotate(angles=(-15,15)),
                ExpandBorder(size=(368,368),resize=True),
                RandomResizedCrop(size=(336, 336)),
                RandomHflip(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': Compose([
                ExpandBorder(size=(336,336),resize=True),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    elif  opt.net == "resnet101":
        data_transforms = {
            'train': Compose([
                RandomRotate(angles=(-15,15)),
                ExpandBorder(size=(396,396),resize=True),
                RandomResizedCrop(size=(368, 368)),
                RandomHflip(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': Compose([
                ExpandBorder(size=(368,368),resize=True),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    elif opt.net == "inceptionv4" or opt.net == "xception" :
        data_transforms = {
            'train': Compose([
                RandomRotate(angles=(-15,15)),
                ExpandBorder(size=(428,428),resize=True),
                RandomResizedCrop(size=(396, 396)),
                RandomHflip(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': Compose([
                ExpandBorder(size=(396,396),resize=True),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    data_set = {}
    data_set['train'] = dataset(anno_pd=train_pd,
                               transforms=data_transforms["train"],
                               )
    data_set['val'] = dataset(anno_pd=val_pd,
                               transforms=data_transforms["val"],
                               )
    dataloader = {}
    dataloader['train']=torch.utils.data.DataLoader(data_set['train'], batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=2*opt.n_cpu,collate_fn=collate_fn)
    dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=8,
                                                   shuffle=True, num_workers=opt.n_cpu,collate_fn=collate_fn)
    '''model'''
    if opt.net == "resnet50":
        model =resnet50(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model.fc = torch.nn.Linear(model.fc.in_features,61)
    elif opt.net == "resnet101":
        model =resnet101(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model.fc = torch.nn.Linear(model.fc.in_features,61)
    elif opt.net == "xception":
        model =pdr_xception(61)
    elif opt.net == "inceptionv4":
        model =inceptionv4(61)
    if opt.resume:
        logging.info('resuming finetune from %s'%opt.resume)
        model.load_state_dict(torch.load(opt.resume))
    model= torch.nn.DataParallel(model)
    model = model.cuda()
    if opt.optim =="sgd":
        optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-5)
    elif opt.optim =="adam":
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)

    if opt.loss =="CrossEntropyLoss":
        criterion = CrossEntropyLoss()
    else:
        criterion = FocalLoss(class_num=61)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    train(model,
          epoch_num=opt.epochs,
          start_epoch=opt.start_epoch,
          optimizer=optimizer,
          criterion=criterion,
          exp_lr_scheduler=exp_lr_scheduler,
          data_set=data_set,
          data_loader=dataloader,
          save_dir=save_dir,
          print_inter=opt.print_interval,
          val_inter=opt.save_checkpoint_val_interval)