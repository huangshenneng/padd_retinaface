from __future__ import print_function
import os
import os.path
import sys
import paddle
from paddle.io import Dataset
import cv2
import numpy as np
from paddle.vision.transforms import ToTensor
from PIL import Image


import tarfile



class WiderFaceDetection(Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

        # 增加多个目标的图片的采样
        time_choose=3
        new_words=[]
        new_imgs_path=[]
        for index in range(len(self.words)):
            lbs=self.words[index]
            nums_lbs=len(lbs)
            if nums_lbs>30:
                imgs_path = self.imgs_path[index]
                for i  in range(time_choose):
                    new_words.append(lbs)
                    new_imgs_path.append(imgs_path)

        self.words+=new_words
        self.imgs_path+=new_imgs_path





    def __getitem__(self, index):
        # index = 9475
        # print('index  ',index)

        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4]<0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        #
        # print( 'inside target  ',target)
        # 把数据合并到数据中
        np_t=np.zeros((1,img.shape[1]*img.shape[0]))
        max_size=img.shape[1]*img.shape[0]
        num_gt=target.shape[0]
        all_gt_num=num_gt*15+5
        if all_gt_num > max_size:
            num_gt = num_gt-1

        np_t[0][0]=num_gt
        # print('num ',num_gt)
        for i in range(num_gt):
            np_t[:,i*15+1:(i+1)*15+1]=target[i]

        np_t = np.reshape(np_t, (1, img.shape[1],img.shape[0]))
        image = Image.fromarray(img.astype('uint8'))
        img = ToTensor()(image)
        return img,np_t



    def __len__(self):
        return len(self.imgs_path)

    def get_spec_index(self,epoch_index):
        self.epoch_index=epoch_index

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if tup.shape[0]==3:
                tup=tup[np.newaxis,:,:,:]
                imgs.append(tup)
            else:
                # annos = paddle.to_tensor(tup,dtype='float32')
                targets.append(tup)
    imgs=np.concatenate(imgs, 0)
    # return (imgs, targets)
    return imgs


