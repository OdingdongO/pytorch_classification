#coding=utf-8
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.AL_defect_dataset import collate_fn, dataset,get_image_pd
import torch
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50,resnet101
from models.multiscale_resnet import multiscale_resnet
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train_util import train, trainlog
from  torch.nn import CrossEntropyLoss
import logging
from dataset.data_aug import *
import sys
import argparse
from dataset.AL_defect_dataset import defect_label

reload(sys)
sys.setdefaultencoding('utf8')
'''
https://tianchi.aliyun.com/competition/information.htm?spm=5176.11165268.5678.2.7d4910c5YfrwzY&raceId=231682
'''
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='size of each image batch')

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--checkpoint_dir', type=str, default='/media/hszc/model/detao/models/AL/resnet50', help='directory where model checkpoints are saved')
parser.add_argument('--cuda_device', type=str, default="2,3", help='whether to use cuda if available')
parser.add_argument('--net', dest='net',type=str, default='resnet50',help='resnet101,resnet50')
parser.add_argument('--resume', type=str, default=None, help='path to resume weights file')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')

parser.add_argument('--save_checkpoint_val_interval', type=int, default=300, help='interval between saving model weights')
parser.add_argument('--print_interval', type=int, default=100, help='interval between print log')
parser.add_argument('--img_root_train', type=str, default= "/media/hszc/model/detao/data/guangdong/guangdong_round1_train2_2018091622/瑕疵样本", help='whether to img root')

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

    all_pd=  get_image_pd(opt.img_root_train)
    train_pd,val_pd = train_test_split(all_pd, test_size=0.1, random_state=53,stratify=all_pd["label"])

    print(val_pd.shape)

    '''数据扩增'''
    data_transforms = {
        'train': Compose([
            Resize(size=(640, 640)),
            FixRandomRotate(bound='Random'),
            RandomHflip(),
            RandomVflip(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': Compose([
            Resize(size=(640, 640)),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_set = {}
    data_set['train'] = dataset( anno_pd=train_pd, transforms=data_transforms["train"])
    data_set['val'] = dataset( anno_pd=val_pd, transforms=data_transforms["val"])

    dataloader = {}
    dataloader['train']=torch.utils.data.DataLoader(data_set['train'], batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=2*opt.n_cpu,collate_fn=collate_fn)
    dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=4,
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

    if opt.resume:
        model.eval()
        logging.info('resuming finetune from %s' % opt.resume)
        try:
            model.load_state_dict(torch.load(opt.resume))
            model = torch.nn.DataParallel(model)
        except KeyError:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(opt.resume))
    else:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-5)
    criterion = CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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