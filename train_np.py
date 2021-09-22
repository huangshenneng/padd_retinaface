#coding: utf-8
from __future__ import print_function
import os
# import torch
# import torch.optim as optim
import paddle.optimizer as optim
# import torch.backends.cudnn as cudnn
import argparse
# import torch.utils.data as data
from paddle.io import Dataset, BatchSampler, DataLoader

from data import WiderFaceDetection,WiderFaceDetection_np, detection_collate, preproc, cfg_mnet, cfg_re50

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
# parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
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

cfg['batch_size']=3
rgb_mean = (104, 117, 123) # bgr order
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
paddle.save(net.state_dict(),'model.pdparams')
bb=paddle.load('model.pdparams')
net.load_dict(paddle.load('model.pdparams'))
print("Printing net...")
# print(net)

paddle.device.set_device("gpu")


# cudnn.benchmark = True


optimizer = optim.SGD(parameters=net.parameters(), learning_rate=initial_lr,  weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
# with torch.no_grad():
priors = priorbox.forward()
priors = priors.cuda()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset_train = WiderFaceDetection_np( training_dataset,preproc(img_dim, rgb_mean))


    len_data=dataset_train.get_length()
    batch_size=2
    for epoch in range(1000):
        print('epoch  ',epoch)
        # for iteration, (images, targets) in enumerate(trainData()):
        index_data=np.arange(len_data)
        np.random.shuffle(index_data)
        iter_batch=0
        list_img=[]
        list_tg=[]
        for iteration, (index) in enumerate(index_data):
            if iteration>batch_size*1:
                break
            # print('iteration  ',iteration)

            load_t0 = time.time()
            images,target=dataset_train.getData(index)
            if images.shape[-1]<10:
                continue
            list_img.append(images)
            list_tg.append( paddle.to_tensor(target, dtype='float32').cuda() )
            iter_batch+=1
            if iter_batch<batch_size:
                continue
            # 合并形成batch
            image_pd=np.concatenate(list_img,0)

            images = paddle.to_tensor(image_pd, dtype='float32')
            images = images.cuda()
            images=images/128.


            # forward
            out = net(images)

            # backprop

            loss_l, loss_c, loss_landm = criterion(out, priors, list_tg)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            load_t1 = time.time()
            batch_time = load_t1 - load_t0

            print('Epoch:{}/{} || Epochiter: {}  Loc: {:.4f}    Cla: {:.4f} Landm: {:.4f} || LR: {:.8f}'
                  .format(epoch, max_epoch, iteration, loss_l.item(), loss_c.item(), loss_landm.item(), initial_lr, batch_time))

            iter_batch = 0
            list_img = []
            list_tg = []
            del images, target

        # net.save(save_folder + cfg['name'] + '_Final',False)
        paddle.save(net.state_dict(), save_folder + 'Final_Retinaface.pdparams')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
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
