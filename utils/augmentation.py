"""
some functions for data augmentation
"""
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import torch.nn.functional as F

def resize(imgs, size) :
    """resize images in Tensor"""
    return F.interpolate(imgs, size=size, mode='bilinear', align_corners=True)

def data_augmentation(img, bounding_boxes, labels) :
    """
    Enhance the data with imgaug
    Largely inspired by https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html
    :param img: single image
    :param bounding_boxes: the list of bounding boxes
    :return:
    """
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], label=label) for bbox, label in zip(bounding_boxes, labels)
    ], shape=img.shape)
    seq = iaa.Sequential([
        # Blur each image with varying strength using
        # gaussian blur (sigma between 0 and 3.0),
        iaa.GaussianBlur((0, 3.0)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Sometimes(
            p=0.5,
            then_list=iaa.Multiply((0.8, 1.2), per_channel=0.2)
        ),
        # Increase saturation
        iaa.Sometimes(
            p=0.5,
            then_list=iaa.MultiplySaturation(mul=(0.5, 1.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.Sometimes(
            p=0.5,
            then_list=iaa.LinearContrast((0.75, 1.5)),
        ),
        iaa.HistogramEqualization(),
        # horizontal flips
        iaa.Fliplr(0.5),
        # random crops
        iaa.Crop(percent=(0, 0.1)),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Sometimes(
            p=0.5,
            then_list=iaa.Affine(
                rotate=(-15, 15),
            ),
        )
    ], random_order=True)
    seq_det = seq.to_deterministic()
    # Augment BBs and images.
    image_aug, bbs_aug = seq_det(image=img, bounding_boxes=bbs)
    bbs_aug = bbs_aug.clip_out_of_image()
    bboxes_aug = list()
    height, width, _ = image_aug.shape
    for i in range(len(bbs_aug.bounding_boxes)) :
        bboxes_aug.append([
            bbs_aug.bounding_boxes[i].label,
            bbs_aug.bounding_boxes[i].x1, bbs_aug.bounding_boxes[i].y1,
            bbs_aug.bounding_boxes[i].x2, bbs_aug.bounding_boxes[i].y2
        ])
    return image_aug, np.array(bboxes_aug, dtype=np.float32)