#coding=utf8
from __future__ import division
import torch
import os,time,datetime
from torch.autograd import Variable
import logging
import numpy as np
from math import ceil

def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def train(model,
          epoch_num,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,
          data_set,
          data_loader,
          save_dir,
          print_inter=200,
          val_inter=3500
          ):

    step = -1
    for epoch in range(start_epoch,epoch_num):
        # train phase
        exp_lr_scheduler.step(epoch)
        model.train(True)  # Set model to training mode

        for batch_cnt, data in enumerate(data_loader['train']):

            step+=1
            model.train(True)
            # print data
            inputs, labels = data

            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            if isinstance(outputs, list):
                loss = criterion(outputs[0], labels)
                loss += criterion(outputs[1], labels)
                outputs=outputs[0]
            else:
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            # batch loss
            if step % print_inter == 0:
                _, preds = torch.max(outputs, 1)

                batch_corrects = torch.sum((preds == labels)).data[0]
                batch_acc = batch_corrects / (labels.size(0))

                logging.info('%s [%d-%d] | batch-loss: %.3f | acc@1: %.3f'
                             % (dt(), epoch, batch_cnt, loss.data[0], batch_acc))


            if step % val_inter == 0:
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                # val phase
                model.train(False)  # Set model to evaluate mode

                val_loss = 0
                val_corrects = 0
                val_size = ceil(len(data_set['val']) / data_loader['val'].batch_size)

                t0 = time.time()

                for batch_cnt_val, data_val in enumerate(data_loader['val']):
                    # print data
                    inputs,  labels = data_val

                    inputs = Variable(inputs.cuda())
                    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

                    # forward
                    outputs = model(inputs)
                    if isinstance(outputs, list):
                        loss = criterion(outputs[0], labels)
                        loss += criterion(outputs[1], labels)
                        outputs = outputs[0]

                    else:
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # statistics
                    val_loss += loss.data[0]
                    batch_corrects = torch.sum((preds == labels)).data[0]
                    val_corrects += batch_corrects

                val_loss = val_loss / val_size
                val_acc = 1.0 * val_corrects / len(data_set['val'])

                t1 = time.time()
                since = t1-t0
                logging.info('--'*30)
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())

                logging.info('%s epoch[%d]-val-loss: %.4f ||val-acc@1: %.4f ||time: %d'
                             % (dt(), epoch, val_loss, val_acc, since))

                # save model
                save_path = os.path.join(save_dir,
                        'weights-%d-%d-[%.4f].pth'%(epoch,batch_cnt,val_acc))
                torch.save(model.state_dict(), save_path)
                logging.info('saved model to %s' % (save_path))
                logging.info('--' * 30)


