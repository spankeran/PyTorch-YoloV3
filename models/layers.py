#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.bbox

class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()
    
class YoloLayer(nn.Module):
    """
    Generate YOLO output
    return x shape: [batch_size * (3*H*W) * (4 + 1 + class_num)]
           priori_boxes shape : [(3*H*W) * 4]
    """

    def __init__(self, anchors, per_cell_anchor_num=3):
        super(YoloLayer, self).__init__()
        self.anchors = torch.from_numpy(anchors).float()
        self.per_cell_anchor_num = per_cell_anchor_num

    def forward(self, x, input_dim):
        device = x.device
        batch_size, map_size = x.size(0), x.size(2)
        scale = input_dim // map_size
        scaled_anchors = self.anchors.to(device) / scale
        priori_boxes   = x.new_zeros(size=(3, map_size, map_size, 4), requires_grad=False)

        # transfer shape from batch_size * channel * N * N to batch_size * 3 * N * N * (4 + 1 + class_num)
        x = x.view(batch_size, 3, -1, map_size, map_size).permute(0, 1, 3, 4, 2).contiguous()
        offset_x = torch.arange(map_size).repeat(map_size, 1).view(1, 1, map_size, map_size).float().to(device)
        offset_y = torch.arange(map_size).repeat(map_size, 1).t().view(1, 1, map_size, map_size).float().to(device)
        offset = torch.stack([offset_x, offset_y], dim=-1)

        x[..., :2] = x[..., :2].sigmoid() + offset
        x[..., 2:4] = x[..., 2:4].exp() * scaled_anchors.view(1, 3, 1, 1, 2)
        x[..., :4] = x[..., :4] / map_size
        priori_boxes[..., :2] = offset.squeeze(1)
        priori_boxes[..., 2:] = scaled_anchors.view(3, 1, 1, 2)
        priori_boxes[..., :4] = priori_boxes[..., :4] / map_size

        x = x.view(batch_size, 3 * map_size * map_size, -1)
        priori_boxes = priori_boxes.view(3 * map_size * map_size, 4)

        return x, priori_boxes, map_size

class YoloLossLayer(nn.Module):
    """
    calculate yolov3 loss
    Args :
        predictions: yolo layer output, shape : [batch_size * priori_box_num * (4 + 1 + class_num)]
        priori_boxes : all priori bounding boxes, shape : [priori_box_num * 4]
        targets: targets in (batch_index, label, cx, cy, w, h) format, shape : [target_box_num, 6]
        train_infos : record infos during training
    """

    def __init__(self, class_num, ignore_thresold, coord_scale, conf_scale, cls_scale):
        super(YoloLossLayer, self).__init__()
        self.class_num   = class_num
        self.ignore_thresold = ignore_thresold
        self.coord_scale = coord_scale
        self.conf_scale  = conf_scale
        self.cls_scale   = cls_scale

    def forward(self, predictions, priori_boxes, featuremap_sizes, input_dim, targets, train_infos):
        total_loss = 0
        batch_size       = predictions.size(0)
        target_boxes_num = targets.size(0)
        priori_boxes_num = priori_boxes.size(0)

        scale_offset,sum = list(), 0
        priori_wh   = predictions.new_zeros(size=(3 * len(featuremap_sizes), 2), requires_grad=False)
        for i in range(priori_wh.size(0)) :
            current_scale = featuremap_sizes[i // 3] ** 2
            scale_offset.append(sum)
            sum += current_scale
            priori_wh[i, :] = priori_boxes[scale_offset[-1], 2:]

        obj_mask    = predictions.new_zeros(size=(batch_size, priori_boxes_num), requires_grad=False).bool()
        noobj_mask  = predictions.new_ones(size=(batch_size, priori_boxes_num), requires_grad=False).bool()
        target_conf = predictions.new_zeros(size=(batch_size, priori_boxes_num), requires_grad=False)
        target_cls  = predictions.new_zeros(size=(batch_size, priori_boxes_num, self.class_num), requires_grad=False)
        iou_scores  = predictions.new_zeros(size=(batch_size, priori_boxes_num), requires_grad=False)
        class_match = predictions.new_zeros(size=(batch_size, priori_boxes_num), requires_grad=False)

        # shape: batch_size * priori_boxes_num * (4, 1, class_num)
        # box form xywh
        pred_boxes = predictions[..., :4]
        pred_conf  = predictions[..., 4]
        pred_cls   = predictions[..., 5:]

        # shape: target_boxes_num * (4, 0, 0)
        # box form xywh
        target_boxes      = targets[..., 2:]
        target_x, target_y= targets[..., 2:4].t()
        target_batch_inds = targets[..., 0].long()
        target_labels     = targets[..., 1].long()


        target2priori_ious    = utils.bbox.bboxes_iou(target_boxes, priori_boxes)
        ignore_priori_mask    = (target2priori_ious > self.ignore_thresold).sum(0) > 0
        best_priori_box_inds  = predictions.new_empty(size=(target_boxes_num,)).long()
        target2priori_whious  = utils.bbox.bboxes_wh_iou(target_boxes[..., 2:], priori_wh)
        for i in range(target_boxes_num) :
            ind = target2priori_whious[i].argmax()
            map_size = featuremap_sizes[ind // 3]
            h, w = int(target_y[i] * map_size), int(target_x[i] * map_size)
            best_priori_box_inds[i] = scale_offset[ind] + ind * map_size ** 2 + h * map_size + w

        positive_pred_boxes = pred_boxes[target_batch_inds, best_priori_box_inds]
        positive_pred_cls   = pred_cls[target_batch_inds, best_priori_box_inds]
        noobj_mask[target_batch_inds, best_priori_box_inds]  = False
        noobj_mask[..., ignore_priori_mask] = False
        target_conf[target_batch_inds, best_priori_box_inds] = 1
        target_cls[target_batch_inds, best_priori_box_inds, target_labels]  = 1
        iou_scores[target_batch_inds, best_priori_box_inds]  = utils.bbox.bboxes_iou(positive_pred_boxes, target_boxes).diag()
        class_match[target_batch_inds, best_priori_box_inds] = (positive_pred_cls.argmax(1) == target_labels).float()

        # GIOU loss
        # Refer to CVPR2019: "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
        giou = utils.bbox.bboxes_GIOU(positive_pred_boxes, target_boxes, xyxy=False)
        coeff_w, coeff_h = target_boxes[:, 2], target_boxes[:, 3]
        coeff = 2. - coeff_w * coeff_h
        giou_loss = (coeff * (1 - giou)).sum()

        positive_pred, negative_pred = pred_conf[obj_mask].sigmoid(), pred_conf[noobj_mask].sigmoid()
        positive_conf_loss = (0.25 * ((1 - positive_pred) ** 2) * F.binary_cross_entropy_with_logits(pred_conf[obj_mask], target_conf[obj_mask], reduction='none')).sum()
        negative_conf_loss = (negative_pred ** 2 * F.binary_cross_entropy_with_logits(pred_conf[noobj_mask], target_conf[noobj_mask], reduction='none')).sum()
        cls_loss  = F.binary_cross_entropy_with_logits(pred_cls[obj_mask], target_cls[obj_mask], reduction='sum')

        giou_loss  = self.coord_scale * giou_loss / batch_size
        conf_loss  = self.conf_scale * (positive_conf_loss + negative_conf_loss) / batch_size
        cls_loss   = self.cls_scale * cls_loss / batch_size
        total_loss += giou_loss + conf_loss + cls_loss

        iou50   = (iou_scores > 0.5).float()
        conf50  = (pred_conf.sigmoid() > 0.5).float()
        avg_iou = iou_scores[obj_mask].mean()
        avg_obj_conf = pred_conf[obj_mask].sigmoid().mean()
        avg_cls_acc  = class_match[obj_mask].float().mean()
        precision    = (class_match * iou50 * conf50).sum() / ((conf50).sum() + 1e-16)
        recall       = (class_match * iou50 * conf50).sum() / (obj_mask.sum() + 1e-16)

        train_infos.update('giou_loss', giou_loss.item())
        train_infos.update('conf_loss', conf_loss.item())
        train_infos.update('cls_loss', cls_loss.item())
        train_infos.update('avg_iou', avg_iou.item())
        train_infos.update('avg_conf', avg_obj_conf.item())
        train_infos.update('avg_cls_acc', avg_cls_acc.item())
        train_infos.update('precision', precision.item())
        train_infos.update('recall', recall.item())

        return total_loss


class YoloPredictLayer(nn.Module):
    """
    generate prediction bounding box
    Args :
        predictions: yolo layer output, shape : [batch_size * priori_box_num * (4 + 1 + class_num)]
    """

    def __init__(self, conf_thresold, nms_thresold) :
        super(YoloPredictLayer, self).__init__()
        self.conf_thresold = conf_thresold
        self.nms_thresold = nms_thresold

    def forward(self, predictions):
        batch_size = predictions.size(0)

        predictions[..., :4] = utils.bbox.xywh2xyxy(predictions[..., :4])
        predictions[..., 4:] = predictions[..., 4:].sigmoid()

        bbox_per_image = [None for _ in range(batch_size)]
        for batch_idx, prediction in enumerate(predictions) :
            # prediction shape : priori_boxes_num * (4 + 1 + class_num)
            prediction = prediction[prediction[..., 4] >= self.conf_thresold]
            if prediction.size(0) == 0:
                continue
            cls_confs, cls_ids = prediction[..., 5:].max(1, keepdim=True)
            current_detections = torch.cat((prediction[:, :5], cls_confs, cls_ids.float()), 1)
            bbox_per_image[batch_idx] = utils.bbox.non_max_suppression(
                current_detections, self.nms_thresold
            )

        return bbox_per_image