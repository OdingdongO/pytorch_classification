#coding=utf-8
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.bd_tiangong_dataset import collate_fn, dataset
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
from dataset.bd_tiangong_dataset import class2label

reload(sys)
sys.setdefaultencoding('utf8')
'''
http://dianshi.baidu.com/dianshi/pc/competition/22/rank'''
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='size of each image batch')

parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--checkpoint_dir', type=str, default='/media/hszc/model/detao/models/bd_tiangong/resnet50', help='directory where model checkpoints are saved')
parser.add_argument('--cuda_device', type=str, default="0", help='whether to use cuda if available')
parser.add_argument('--net', dest='net',type=str, default='resnet50',help='resnet101,resnet50')
parser.add_argument('--resume', type=str, default=None, help='path to resume weights file')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')

parser.add_argument('--save_checkpoint_val_interval', type=int, default=200, help='interval between saving model weights')
parser.add_argument('--print_interval', type=int, default=100, help='interval between print log')
# parser.add_argument('--img_root_train', type=str, default= "/media/hszc/model/detao/data/guangdong/guangdong_round1_train2_2018091622/瑕疵样本", help='whether to img root')

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

    label_csv ="/media/hszc/model/detao/data/dianshi/宽波段数据集-预赛训练集2000/光学数据集-预赛训练集-2000-有标签.csv"
    imgroot="/media/hszc/model/detao/data/dianshi/宽波段数据集-预赛训练集2000/预赛训练集-2000"
    label = pd.read_csv(label_csv,names=["ImageName","class"])
    label["label"]=label["class"].apply(lambda x:class2label[x])
    train_pd,val_pd = train_test_split(label, test_size=0.12, random_state=43,stratify=label["label"])

    print(val_pd.shape)

    '''数据扩增'''
    data_transforms = {
        'train': Compose([
            Resize(size=(256, 256)),
            FixRandomRotate(bound='Random'),
            RandomHflip(),
            RandomVflip(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': Compose([
            Resize(size=(256, 256)),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_set = {}
    data_set['train'] = dataset(imgroot, anno_pd=train_pd, transforms=data_transforms["train"])
    data_set['val'] = dataset(imgroot, anno_pd=val_pd, transforms=data_transforms["val"])

    dataloader = {}
    dataloader['train']=torch.utils.data.DataLoader(data_set['train'], batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=2*opt.n_cpu,collate_fn=collate_fn)
    dataloader['val']=torch.utils.data.DataLoader(data_set['val'], batch_size=4,
                                                   shuffle=True, num_workers=opt.n_cpu,collate_fn=collate_fn)
    '''model'''
    if opt.net == "resnet50":
        model =resnet50(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model.fc = torch.nn.Linear(model.fc.in_features,6)
    elif opt.net == "resnet101":
        model =resnet101(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model.fc = torch.nn.Linear(model.fc.in_features,6)

    if opt.resume:
        model.eval()
        logging.info('resuming finetune from %s' % opt.resume)
        try:
            model.load_state_dict(torch.load(opt.resume))
        except KeyError:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(opt.resume))
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-5)
    criterion = CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

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