import paddle
import numpy as np


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return paddle.concat([boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2], 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return paddle.concat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]

    # aa= paddle.unsqueeze(box_a[:, 2:],axis=1).expand([A, B, 2])
    # bb=paddle.unsqueeze(box_b[:, 2:],axis=0).expand([A, B, 2])
    max_xy = paddle.minimum(
        paddle.unsqueeze(box_a[:, 2:],axis=1).expand([A, B, 2]),
        paddle.unsqueeze(box_b[:, 2:],axis=0).expand([A, B, 2]),
                       )
    min_xy = paddle.maximum(
        paddle.unsqueeze(box_a[:, :2], axis=1).expand([A, B, 2]),
        paddle.unsqueeze(box_b[:, :2], axis=0).expand([A, B, 2]),
    )
    inter = paddle.clip((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    # best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_overlap = paddle.max(overlaps,axis=1, keepdim=True)
    best_prior_idx = paddle.argmax(overlaps,axis=1, keepdim=True)

    # print('best_prior_overlap  ',best_prior_overlap)
    # print('best_prior_idx  ',best_prior_idx)

    # import sys
    # sys.exit()
    # ignore hard gt
    valid_gt_idx = paddle.greater_than(best_prior_overlap,paddle.ones_like(best_prior_overlap)*0.2)
    best_prior_idx_filter = paddle.masked_select(best_prior_idx,valid_gt_idx)


    # 转成numpy格式
    # best_prior_overlap_np=best_prior_overlap.numpy()
    # valid_gt_idx = best_prior_overlap_np[:, 0] >= 0.2
    # best_prior_idx_np=best_prior_idx.numpy()
    # best_prior_idx_filter = best_prior_idx_np[valid_gt_idx, :]

    # best_prior_idx_filter=paddle.to_tensor(best_prior_idx_filter)
    # best_prior_idx=paddle.to_tensor(best_prior_idx)
    if best_prior_idx_filter.shape[0] <= 0:
        del best_prior_idx_filter
        del best_prior_idx
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # 转回paddle格式

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap = paddle.max(overlaps, axis=0, keepdim=True)
    best_truth_idx = paddle.argmax(overlaps, axis=0, keepdim=True).astype('int64')  # 'float32' int64

    best_truth_idx=paddle.squeeze(best_truth_idx,axis=0)
    best_truth_overlap=paddle.squeeze(best_truth_overlap,axis=0)
    best_prior_idx=paddle.squeeze(best_prior_idx,axis=1)

    # print()
    # best_prior_idx_filter=paddle.squeeze(best_prior_idx_filter,axis=1)


    # best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior
    for index in  best_prior_idx_filter:
        index=index.numpy()
        best_truth_overlap[index] = 2.

    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.shape[0]):     # 判别此anchor是预测哪一个boxes
        index2=best_prior_idx[j].numpy()
        best_truth_idx[index2] = j
    matches = truths[best_truth_idx]            # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
    conf = labels[best_truth_idx]               # Shape: [num_priors]      此处为每一个anchor对应的label取出来


    conf[best_truth_overlap < threshold] = 0    # label as background   overlap<0.35的全部作为负样本
    loc = encode(matches, priors, variances)

    # conf_np = conf.cpu().numpy()
    # a, b = np.unique(conf_np, return_counts=True)
    # print('a ,b',(a,b))

    matches_landm = landms[best_truth_idx]
    landm = encode_landm(matches_landm, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    landm_t[idx] = landm

    del best_prior_overlap
    del best_prior_idx
    del best_prior_idx_filter
    del best_truth_idx


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = paddle.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return paddle.concat([g_cxcy, g_wh], 1)  # [num_priors,4]

def encode_landm(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 10].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded landm (tensor), Shape: [num_priors, 10]
    """

    # dist b/t match center and prior's center
    matched = paddle.reshape(matched, [matched.shape[0], 5, 2])

    # a=paddle.unsqueeze(x=priors[:, 0], axis=1)
    # b=paddle.expand( paddle.unsqueeze(x=priors[:, 0],axis=1),[matched.size(0), 5])
    # priors_cx =paddle.unsqueeze(  paddle.expand( paddle.unsqueeze(x=priors[:, 0],axis=1),[matched.size(0), 5]),axis=2)

    priors_cx =paddle.unsqueeze(  paddle.expand( paddle.unsqueeze(x=priors[:, 0],axis=1),[matched.shape[0], 5]),axis=2)
    priors_cy =paddle.unsqueeze(  paddle.expand( paddle.unsqueeze(x=priors[:, 1],axis=1),[matched.shape[0], 5]),axis=2)
    priors_w =paddle.unsqueeze(  paddle.expand( paddle.unsqueeze(x=priors[:, 2],axis=1),[matched.shape[0], 5]),axis=2)
    priors_h =paddle.unsqueeze(  paddle.expand( paddle.unsqueeze(x=priors[:, 3],axis=1),[matched.shape[0], 5]),axis=2)


    # priors_cx = priors[:, 0].unsqueeze(1).expand(matched.shape[0], 5).unsqueeze(2)
    # priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    # priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    # priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)

    priors = paddle.concat([priors_cx, priors_cy, priors_w, priors_h], axis=2)
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    # g_cxcy /= priors[:, :, 2:]
    g_cxcy =paddle.reshape(g_cxcy,[g_cxcy.shape[0], -1])
    # return target for smooth_l1_loss
    return g_cxcy


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = paddle.concat([
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * paddle.exp(loc[:, 2:] * variances[1])], 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = paddle.concat([priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ], axis=1)
    return landms


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.max()
    return paddle.log(paddle.sum(paddle.exp(x-x_max), 1, keepdim=True)) + x_max






