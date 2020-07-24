#-*-coding:utf-8-*-
"""
some functions for bounding box
"""

import random
import torch
from torch import Tensor
import matplotlib.pyplot as plt

def xywh2xyxy(bboxes) :
    new_bboxes_min = bboxes[..., :2] - bboxes[..., 2:] / 2
    new_bboxes_max = bboxes[..., :2] + bboxes[..., 2:] / 2
    return torch.cat([new_bboxes_min, new_bboxes_max], dim=-1)

def xyxy2xywh(bboxes) :
    new_bboxes_xy = (bboxes[..., 2:] + bboxes[..., :2]) / 2
    new_bboxes_wh = bboxes[..., 2:] - bboxes[..., :2]
    return torch.cat([new_bboxes_xy, new_bboxes_wh], dim=-1)

def bboxes_area(bboxes) :
    """calculate bboxes area"""
    w = torch.clamp(bboxes[:, 2] - bboxes[:, 0], min=0.)
    h = torch.clamp(bboxes[:, 3] - bboxes[:, 1], min=0.)
    return w * h

def intersect_area(bboxes_a, bboxes_b) :
    """calculate intersect area of bboxes_a and bboxes_b"""
    num_a, num_b = bboxes_a.size(0), bboxes_b.size(0)
    min_xy = torch.max(
        bboxes_a[:, :2].unsqueeze(1).expand(num_a, num_b, 2),
        bboxes_b[:, :2].unsqueeze(0).expand(num_a, num_b, 2),
    )
    max_xy = torch.min(
        bboxes_a[:, 2:].unsqueeze(1).expand(num_a, num_b, 2),
        bboxes_b[:, 2:].unsqueeze(0).expand(num_a, num_b, 2),
    )
    inter = torch.clamp(max_xy - min_xy, min=0.)
    return inter[..., 0] * inter[..., 1]

def union_area(bboxes_a, bboxes_b) :
    """calculate union area of bboxes_a and bboxes_b"""
    num_a, num_b = bboxes_a.size(0), bboxes_b.size(0)
    area_a = bboxes_area(bboxes_a).unsqueeze(1).expand(num_a, num_b)
    area_b = bboxes_area(bboxes_b).unsqueeze(0).expand(num_a, num_b)
    union_area = area_a + area_b
    return union_area

def bboxes_iou(bboxes_a, bboxes_b, xyxy=False) :
    """calculate jaccard index"""
    if not xyxy :
        bboxes_a = xywh2xyxy(bboxes_a)
        bboxes_b = xywh2xyxy(bboxes_b)
    inter = intersect_area(bboxes_a, bboxes_b)
    union = union_area(bboxes_a, bboxes_b) - inter + 1e-16
    return inter / union

def bboxes_wh_iou(bboxes_a, bboxes_b) :
    num_a, num_b = bboxes_a.size(0), bboxes_b.size(0)
    inter_wh = torch.min(
        bboxes_a.unsqueeze(1).expand(num_a, num_b, 2),
        bboxes_b.unsqueeze(0).expand(num_a, num_b, 2)
    )
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = (bboxes_a[..., 0] * bboxes_a[..., 1]).unsqueeze(1).expand(num_a, num_b)
    area_b = (bboxes_b[..., 0] * bboxes_b[..., 1]).unsqueeze(0).expand(num_a, num_b)
    union = area_a + area_b - inter + 1e-16
    return inter / union

def bboxes_GIOU(bboxes_a, bboxes_b, xyxy=False) :
    """calculate GIOU of bounding boxes
    Refer to CVPR2019: "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression" """
    if not xyxy :
        bboxes_a = xywh2xyxy(bboxes_a)
        bboxes_b = xywh2xyxy(bboxes_b)
    inter_min_xy = torch.max(
        bboxes_a[:, :2],
        bboxes_b[:, :2]
    )
    inter_max_xy = torch.min(
        bboxes_a[:, 2:],
        bboxes_b[:, 2:]
    )
    inter_bboxes = torch.clamp(inter_max_xy - inter_min_xy, min=0.)
    inter = inter_bboxes[:, 0] * inter_bboxes[:, 1]
    bboxes_a_wh = torch.clamp(bboxes_a[:, 2:] - bboxes_a[:, :2], min=0.)
    bboxes_b_wh = torch.clamp(bboxes_b[:, 2:] - bboxes_b[:, :2], min=0.)
    bboxes_a_area = bboxes_a_wh[:, 0] * bboxes_a_wh[:, 1]
    bboxes_b_area = bboxes_b_wh[:, 0] * bboxes_b_wh[:, 1]
    union = (bboxes_a_area) + (bboxes_b_area) - inter + 1e-16
    #calculate the minium bbox to cover bbox_a and bbox_b
    bboxes_c_min_xy = torch.min(
        bboxes_a[:, :2],
        bboxes_b[:, :2]
    )
    bboxes_c_max_xy = torch.max(
        bboxes_a[:, 2:],
        bboxes_b[:, 2:]
    )
    bboxes_c_wh = torch.clamp(bboxes_c_max_xy - bboxes_c_min_xy, min=0.)
    bboxes_c_area = bboxes_c_wh[..., 0] * bboxes_c_wh[..., 1] + 1e-16
    iou = inter / union
    return iou - (bboxes_c_area - union) / bboxes_c_area

def non_max_suppression(current_detections, nms_thresold) :
    scores = current_detections[..., 4] * current_detections[..., 5]
    current_detections = current_detections[scores.argsort(descending=True)]
    ret_bboxes, remain_bboxes = None, list()
    while current_detections.size(0) :
        iou_mask = bboxes_iou(current_detections[0, :4].unsqueeze(0),
                                        current_detections[:, :4], xyxy=True) > nms_thresold
        same_cls_mask = current_detections[0, -1] == current_detections[:, -1]
        invalid_mask = (iou_mask.view(-1)) & same_cls_mask
        weight = current_detections[invalid_mask, 4:5]
        current_detections[0, :4] = (weight * current_detections[invalid_mask, :4]).sum(0) / (weight.sum())
        remain_bboxes += [current_detections[0]]
        current_detections = current_detections[~invalid_mask]

    if remain_bboxes:
        ret_bboxes = torch.stack(remain_bboxes)

    return ret_bboxes

def plt_bboxes(img, classes, scores, bboxes, save=False, save_path=None, linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!"""
    plt.axis('off')
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(len(classes)):
        class_name = classes[i]
        score = scores[i]
        if class_name not in colors:
            colors[class_name] = (random.random(), random.random(), random.random())
        colors = colors
        xmin = int(bboxes[i, 0] * width)
        ymin = int(bboxes[i, 1] * height)
        xmax = int(bboxes[i, 2] * width)
        ymax = int(bboxes[i, 3] * height)
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                ymax - ymin, fill=False,
                                edgecolor=colors[class_name],
                                linewidth=linewidth)
        plt.gca().add_patch(rect)
        plt.gca().text(xmin, ymin + 20,
                        '{:s} | {:.3f}'.format(class_name, score),
                        bbox=dict(facecolor=colors[class_name], alpha=0.5),
                        fontsize=10, color='white')
    if save and save_path :
        plt.savefig(save_path)

    plt.show()
    plt.close()