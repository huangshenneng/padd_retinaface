#coding: utf-8
from __future__ import print_function
import os

import paddle.optimizer as optim
import argparse
from paddle.io import Dataset, BatchSampler, DataLoader

from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox

import time
import datetime
import numpy as np
import math
from models.retinaface import RetinaFace
import sys
import paddle

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
# parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

cfg['batch_size']=2


rgb_mean = (104, 117,  123) # bgr order

num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)

print("Printing net...")


paddle.device.set_device("gpu")





optimizer = optim.SGD(parameters=net.parameters(), learning_rate=initial_lr,  weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))

priors = priorbox.forward().detach()
priors = priors.cuda()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset_train = WiderFaceDetection( training_dataset,preproc(img_dim, rgb_mean))
    epoch_size = math.ceil(len(dataset_train) / batch_size)
    max_iter = max_epoch * epoch_size



    train_loader=paddle.io.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
    log_train = open('train.txt', 'a+')
    print('------------------------- train : batchsize %d  ---------------------------' % batch_size)
    print('------------------------- train : batchsize %d  ---------------------------' % batch_size, file=log_train)
    show_time=10
    accumulation_steps = 12
    for epoch in range(1000):
        print('epoch  ',epoch)

        epoch_image_index=[]
        for iteration, (images,target) in enumerate(train_loader()):

            load_t0 = time.time()
            targets=[]

            for tg in target:
                tg=paddle.reshape(tg,[1,-1])[0]
                num=tg[0].numpy().astype(np.int)[0]
                lbs=paddle.ones([num,15],dtype='float64')
                for i in range(num):
                    lbs[i]=tg[i*15+1:(i+1)*15+1]
                lbs=lbs.astype('float32')
                targets.append(lbs)

            # print('images  shape',images.shape)
            # #forward
            out = net(images)


            loss_l, loss_c, loss_landm = criterion(out, priors, targets)

            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss /= accumulation_steps
            loss.backward()
            if ((iteration + 1) % accumulation_steps) == 0:
                optimizer.step()
                optimizer.clear_grad()


            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - iteration))
            if iteration%show_time==0:
                print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                      .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                      epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), initial_lr, batch_time, str(datetime.timedelta(seconds=eta))))
        #
                print(
                    'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                    .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                            epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), initial_lr,
                            batch_time, str(datetime.timedelta(seconds=eta))) ,file=log_train)
                log_train.close()
                log_train = open('train.txt', 'a+')

            del images
            for lbs in targets:
                del lbs
            del targets

        paddle.save(net.state_dict(), save_folder + '%d_Retinaface.pdparams'%epoch)

    paddle.save(net.state_dict(), save_folder + 'Final_Retinaface.pdparams')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate

    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
