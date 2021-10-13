
import paddle.nn as nn
import paddle.nn.functional as F

from utils.box_utils import match, log_sum_exp
import numpy as np
from data import cfg_mnet
import sys

GPU = cfg_mnet['gpu_train']
import paddle

data_type_ = 'float32'
class MultiBoxLoss:
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.


            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.shape[0]
        num_priors = (priors.shape[0])

        # match priors (default boxes) and ground truth boxes

        loc_t = paddle.zeros(shape=[num, num_priors, 4])
        landm_t = paddle.zeros(shape=[num, num_priors, 10])
        conf_t = paddle.zeros(shape=[num, num_priors], dtype='float32')

        for idx in range(num):
            truths = targets[idx][:, :4]
            labels = targets[idx][:, -1]

            landms = targets[idx][:, 4:14]
            defaults = priors
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        zeros = paddle.to_tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.astype(data_type_).sum(1, keepdim=True)
        N1 = max(num_pos_landm.sum().astype('float32'), 1)
        pos_idx1 = paddle.expand_as(  paddle.unsqueeze(pos1,axis=pos1.dim()),landm_data)


        if N1!=1:
            landm_p = paddle.reshape(paddle.masked_select(landm_data, pos_idx1), [-1, 10])
            landm_t = paddle.reshape(paddle.masked_select(landm_t, pos_idx1), [-1, 10])
            loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        else:
            loss_landm = paddle.to_tensor(0,dtype='float32').cuda()

        pos = conf_t != zeros
        conf_t[pos] = 1


        pos_idx = paddle.expand_as( paddle.unsqueeze(pos,pos.dim()),loc_data)
        loc_p = paddle.reshape(  paddle.masked_select(loc_data,pos_idx),[-1, 4])
        loc_t =  paddle.reshape(  paddle.masked_select(loc_t,pos_idx),[-1, 4])
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf =paddle.reshape( conf_data,[-1, self.num_classes])
        loss_c = log_sum_exp(batch_conf) - gather(batch_conf,conf_t)

        # Hard Negative Mining
        loss_c = paddle.reshape(loss_c,[num, -1])
        loss_idx = paddle.argsort(loss_c,axis=1, descending=True)
        idx_rank = paddle.argsort(loss_idx,axis=1)
        num_pos = pos.astype(data_type_).sum(1, keepdim=True)
        num_neg = paddle.clip(self.negpos_ratio*num_pos, max=pos.shape[1]-1)
        neg = idx_rank < paddle.expand_as(num_neg,idx_rank)



        # Confidence Loss Including Positive and Negative Examples
        pos_idx = paddle.expand_as(  paddle.unsqueeze(pos,2),conf_data).astype(data_type_)
        neg_idx = paddle.expand_as(  paddle.unsqueeze(neg,2),conf_data).astype(data_type_)

        neg_pos_idx=paddle.greater_than( paddle.add(pos_idx,neg_idx),paddle.zeros_like(neg_idx))
        bb=paddle.add(pos.astype(data_type_),neg.astype(data_type_))
        neg_pos=paddle.greater_than( bb,paddle.zeros_like(neg).astype(data_type_))

        conf_p = paddle.reshape( paddle.masked_select(conf_data,neg_pos_idx),[-1,self.num_classes])
        targets_weighted = paddle.unsqueeze( paddle.masked_select(conf_t,neg_pos),axis=1).astype(data_type_)

        del loss_c
        loss_c = F.cross_entropy(conf_p, targets_weighted.astype('int64'), reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.sum().astype('float32'), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        del loc_t
        del conf_t
        del loc_p
        del pos_idx1
        del num_pos_landm
        del zeros
        del landm_t

        del pos_idx
        del neg_idx
        del bb
        del targets_weighted
        del num_pos

        if N1 != 1:
            del landm_p


        return loss_l, loss_c, loss_landm

def gather(batch_conf,conf_t):
    conf_t = paddle.reshape(conf_t, [-1, 1]).astype(data_type_)
    conf_t_one = paddle.ones_like(conf_t, dtype=data_type_)
    sub_conf_t = paddle.subtract(conf_t_one, conf_t)
    cat_conf = paddle.concat([sub_conf_t, conf_t], axis=1).astype('bool')
    batch_conf = paddle.masked_select(batch_conf, cat_conf)
    batch_conf = paddle.unsqueeze(batch_conf, axis=1)
    return batch_conf
