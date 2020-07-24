# -*-coding:utf-8-*-
import cv2
import random
import torch
import os.path as osp
import numpy as np

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

import utils.augmentation
import utils.bbox

BASE_DIR = osp.join(osp.dirname(osp.abspath(__file__)), 'dataset')

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

COCO_CLASSES = [
'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', #airplane motorbicyle
'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'pottedplant', 'bed', #potted plant
'dining table', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', #tv
'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

voc_class2labels = {k: v for v, k in enumerate(VOC_CLASSES)}
coco_class2labels = {k: v for v, k in enumerate(COCO_CLASSES)}

def VOC_annotation_transform(target: ET.Element, keep_difficult=False):
    """
    transforms VOC dataset annotation
    :param target: (ET.Element) annotation for image
    :param keep_difficult: (boolean) whether to keep hard objects
    :return: (numpy.ndarray) just like [[xmin, ymin, xmax, ymax, label],...]
    """
    width, height = int(target.find('size').find('width').text), \
                    int(target.find('size').find('height').text)
    res = []
    for obj in target.iter('object'):
        try:
            is_difficult = int(obj.find('difficult').text)
        except:
            is_difficult = 0
        if not keep_difficult and is_difficult:
            continue

        class_name = obj.find('name').text.lower().strip()
        BBox = obj.find('bndbox')
        label = voc_class2labels[class_name]
        bndbox = list()
        bndbox.append(label)
        for cor in ['xmin', 'ymin', 'xmax', 'ymax']:
            cur_cor = int(BBox.find(cor).text) - 1
            bndbox.append(cur_cor)
        res += [bndbox]

    return np.array(res, dtype=np.float32)

class VOCDataset(Dataset):
    """
    Pascal VOC Detection Dataset. Refer to http://host.robots.ox.ac.uk/pascal/VOC/
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        transform (callable, optional): A function/transform that  takes in an PIL image and annotation
            and returns a transformed version.
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root=BASE_DIR,
                 setname=(('2007', 'trainval'), ('2012', 'trainval'), ('2007', 'test')),
                 transform=None,
                 input_size=416,
                 multi_scale=True):

        self.root = root
        self.setname = setname
        self.dataset_path = osp.join(root, 'VOC%s_%s')
        self.transform = transform
        self.target_transform = VOC_annotation_transform
        self.class_names = VOC_CLASSES
        self.input_size = input_size
        self.multi_scale = multi_scale

        self.max_input_size = input_size + 3 * 32
        self.min_input_size = input_size - 3 * 32

        self.batch_count = 0
        self.image_path = osp.join(self.dataset_path, 'JPEGImages', '%s.jpg')
        self.annotation_path = osp.join(self.dataset_path, 'Annotations', '%s.xml')

        self.data_ids = []
        for (year, train_name) in self.setname:
            dataset_path = self.dataset_path % (year, train_name)
            with open(osp.join(dataset_path, 'ImageSets', 'Main', '%s.txt' % train_name), 'r') as f:
                for line in f.readlines():
                    self.data_ids.append((year, train_name, line.strip()))

    def __len__(self):
        return self.data_ids.__len__()

    def __getitem__(self, index):
        cur_id = self.data_ids[index]
        # height , width , channel in BGR
        img = cv2.imread(self.image_path % cur_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        targets = ET.parse(self.annotation_path % cur_id).getroot()

        # Pad to square
        img, pad = self.pad_to_square(img, 0)
        padded_h, padded_w, _ = img.shape

        targets = self.target_transform(targets)
        # Adjust for added padding
        targets[:, 1:] += np.array(pad)

        if self.transform is not None:
            img, targets = self.transform(img, targets[..., 1:], targets[..., 0])

        # transform to channel, height , width then scaled
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.
        tensor_img = torch.from_numpy(img)
        targets = torch.from_numpy(targets)
        targets[..., 1::2] /= padded_w
        targets[..., 2::2] /= padded_h
        targets[..., 1:] = utils.bbox.xyxy2xywh(targets[..., 1:])

        return tensor_img, targets, cur_id[-1]

    def get_original_image(self, index):
        cur_id = self.data_ids[index]
        # height , width , channel in BGR
        img = cv2.imread(self.image_path % cur_id)
        return img

    def get_original_targets(self, index):
        cur_id = self.data_ids[index]
        targets = ET.parse(self.annotation_path % cur_id).getroot()
        targets = self.target_transform(targets)
        return targets

    def get_class_num(self):
        return len(VOC_CLASSES)

    def pad_to_square(self, img, pad_value) :
        """Pad to square"""
        h, w, c = img.shape
        dim_diff = np.abs(h - w)
        # (upper or left) padding and (lower or right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = [(0, 0), (pad1, pad2), (0, 0)] if h >= w else [(pad1, pad2), (0, 0), (0, 0)]
        # Add padding
        img = np.pad(img, pad, mode='constant', constant_values=(pad_value, pad_value))

        return img, (pad1, 0, pad1, 0) if h >= w else (0, pad1, 0, pad2)

    def collate_fn(self):
        def collate_fn(batch):
            imgs, targets, img_ids = list(), list(), list()
            for idx, (img, target, img_id) in enumerate(batch):
                imgs.append(img.unsqueeze(0))
                img_ids.append(img_id)
                if target.size(0) == 0 :
                    continue
                sample_idx = target.new_full(size=(target.size(0), 1), fill_value=idx)
                target = torch.cat([sample_idx, target], dim=1)
                targets.append(target)

            if self.multi_scale and self.batch_count % 10 == 0:
                self.input_size = random.choice(range(self.min_input_size, self.max_input_size + 1, 32))
            imgs = [utils.augmentation.resize(img, self.input_size) for img in imgs]
            imgs = torch.cat(imgs, dim=0)
            targets = torch.cat(targets, dim=0)
            self.batch_count += 1
            return imgs, targets, img_ids

        return collate_fn