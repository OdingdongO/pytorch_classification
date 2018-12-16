#coding=utf-8
import os
import numpy as np
import pandas as pd
from dataset.bd_tiangong_dataset import dataset, collate_fn,class2label,label2class
import torch
from torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from math import ceil
from  torch.nn.functional import softmax
from dataset.data_aug import *
import glob
from models.resnet import resnet50,resnet101
from models.inception_resnet_v2 import inceptionresnetv2
from models.densenet import densenet169,densenet121
import argparse
from dataset.bd_tiangong_dataset import class2label
from torch import nn
'''
http://dianshi.baidu.com/dianshi/pc/competition/22/rank'''
parser = argparse.ArgumentParser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser.add_argument('--net', dest='net',type=str, default='resnet50',help='densent121,resnet101,resnet50,inceptionv4,densent169,inceptionresnetv2')
parser.add_argument('--resume', type=str, default="/home/detao/Videos/detao/code/pytorch_classification/predict/"
                                                  "model_weights/resnet50_unknow/best_weigths_[1.0000000000000002e-06].pth")
# parser.add_argument('--resume', type=str, default="/home/detao/Videos/detao/code/pytorch_classification/predict/"
#                                                   "model_weights/densent169_unknow/best_weigths_[1.0000000000000002e-06].pth")
# parser.add_argument('--resume', type=str, default="/home/detao/Videos/detao/code/pytorch_classification/predict/"
#                                                   "model_weights/densenet121_unknow/best_weigths_[0.0001].pth")
# parser.add_argument('--resume', type=str, default="/home/detao/Videos/detao/code/pytorch_classification/predict/"
#                                                   "model_weights/resnet101_unknow/best_weigths_[0.001].pth")
parser.add_argument('--mode', type=str, default="val", help='val,test')
parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')

parser.add_argument('--save_checkpoint_val_interval', type=int, default=200, help='interval between saving model weights')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
test_transforms= Compose([
    Resize(size=(256, 256)),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
def np2str(arr):
    return ";".join(["%.16f" % x for x in arr])
def str2np(str):
    return np.array([x for x in str.split(";")])
if __name__ == '__main__':
    if opt.mode=="val":
        label_csv = "/home/detao/Videos/detao/data/all/label/9970_b.csv"
        train_root = "/home/detao/Videos/detao/data/all/data/testb"
    else:
        label_csv = "/home/detao/Videos/detao/data/all/label/9990_a.csv"
        train_root = "/home/detao/Videos/detao/data/all/data/testa/testa"
    test_root = "/media/hszc/model/detao/data/dianshi/宽波段数据集-预赛测试集A1000/预赛测试集A-1000"
    val_pd = pd.read_csv(label_csv, names=["ImageName", "class"])
    val_pd["label"] = val_pd["class"].apply(lambda x: class2label[x])
    val_pd["img_path"]=val_pd["ImageName"].apply(lambda x:os.path.join(train_root,x))

    test_list = glob.glob(test_root+"/*.jpg")
    true_test_pb=pd.DataFrame(test_list,columns=["path"])
    true_test_pb["ImageName"]=true_test_pb["path"].apply(lambda x:os.path.basename(x))
    "addFakeLabel"
    true_test_pb['label'] =1

    test_pd =true_test_pb if opt.mode=="test" else val_pd
    rawdata_root =test_root if opt.mode=="test" else train_root

    print(test_pd.head())
    '''data_set&data_loader'''
    data_set = {}
    data_set['test'] = dataset(imgroot=rawdata_root, anno_pd=test_pd,
                                 transforms=test_transforms,
                                 )
    data_loader = {}
    data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=4, num_workers=4,
                                               shuffle=False, pin_memory=True, collate_fn=collate_fn)

    '''model'''
    if opt.net == "resnet50":
        model =resnet50(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model.fc = torch.nn.Linear(model.fc.in_features,6)
    elif opt.net == "resnet101":
        model =resnet101(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model.fc = torch.nn.Linear(model.fc.in_features,6)
    elif opt.net == "inceptionresnetv2":
        model =inceptionresnetv2(num_classes=6)
    elif opt.net == "densent169":
        model =densenet169(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 6)
    elif opt.net == "densent121":
        model =densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 6)
    if opt.resume:
        model.eval()
        print('resuming finetune from %s' % opt.resume)
        try:
            model.load_state_dict(torch.load(opt.resume))
        except KeyError:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(opt.resume))
    model = model.cuda()
    model.eval()

    criterion = CrossEntropyLoss()

    if not os.path.exists('./tiangong/csv'):
        os.makedirs('./tiangong/csv')

    test_size = ceil(len(data_set['test']) / data_loader['test'].batch_size)
    test_preds = np.zeros((len(data_set['test'])), dtype=np.float32)
    true_label = np.zeros((len(data_set['test'])), dtype=np.int)
    test_scores = np.zeros((len(data_set['test']),6), dtype=np.float32)

    idx = 0
    test_loss = 0
    test_corrects = 0
    for batch_cnt_test, data_test in enumerate(data_loader['test']):
        # print data
        print("{0}/{1}".format(batch_cnt_test, int(test_size)))
        inputs, labels = data_test
        inputs = Variable(inputs.cuda())
        labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
        # forward
        outputs = model(inputs)
        # print(torch.sum(outputs.sigmoid()>0.5).cpu().float())
        # num =torch.sum(outputs.sigmoid()>0.288).cpu().float()
        scores=outputs.sigmoid()
        print(scores)
        if isinstance(outputs, list):
            loss = criterion(outputs[0], labels)
            loss += criterion(outputs[1], labels)
            outputs = (outputs[0]+outputs[1])/2
        else:
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        test_loss += loss.data[0]
        batch_corrects = torch.sum((preds == labels)).data[0]
        test_corrects += batch_corrects
        test_preds[idx:(idx + labels.size(0))] = preds
        test_scores[idx:(idx + labels.size(0))] = scores.data.cpu().numpy()
        true_label[idx:(idx + labels.size(0))] = labels.data.cpu().numpy()
        # statistics
        idx += labels.size(0)
    test_loss = test_loss / test_size
    test_acc = 1.0 * test_corrects / len(data_set['test'])
    print('test-loss: %.4f ||test-acc@1: %.4f'
          % (test_loss, test_acc))

    test_pred = test_pd[['ImageName']].copy()
    test_pred['label'] = list(test_preds)
    test_pred['label'] = test_pred['label'].apply(lambda x: label2class[int(x)])
    test_pred['prob'] = list(test_scores)
    test_pred['prob'] = test_pred['prob'].apply(lambda x: np2str(x))
    test_pred[['ImageName',"label","prob"]].to_csv('tiangong/csv/{0}_{1}_prob.csv'.format(opt.net,opt.mode) ,sep=",",
                                                                     header=None, index=False)
    test_pred[['ImageName',"label"]].to_csv('tiangong/csv/{0}_{1}_result.csv'.format(opt.net,opt.mode) ,sep=",",
                                                                     header=None, index=False)