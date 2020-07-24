#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import numpy as np

from models.layers import EmptyLayer, YoloLayer, YoloLossLayer, YoloPredictLayer

def parse_cfg(cfg_file_path) :
    """
    Parse configuration file
    :param cfg_file_path: the path of configuration file
    :return: list of blocks
    """
    block, blocks = dict(), list()
    with open(cfg_file_path, 'r') as file :
        while True :
            line = file.readline()
            if not line :
                break
            line = line.lstrip().rstrip()
            if len(line) == 0 or line[0] == '#' :
                continue
            if line[0] == '[' :
                if len(block) > 0 :
                    blocks.append(block)
                    block = dict()
                block['type'] = line[1:-1]
            else :
                name, value = line.split('=')
                block[name.rstrip()] = value.lstrip()
        blocks.append(block)
    return blocks

class yolo_network(nn.Module) :

    def __init__(self, network_name, class_num,
                 ignore_thresold, conf_thresold, nms_thresold,
                 coord_scale, conf_scale, cls_scale,
                    device) :
        super(yolo_network, self).__init__()
        self.network_name = network_name
        self.class_num    = class_num
        self.ignore_thresold = ignore_thresold
        self.conf_thresold   = conf_thresold
        self.nms_thresold = nms_thresold
        self.coord_scale  = coord_scale
        self.conf_scale = conf_scale
        self.cls_scale  = cls_scale
        self.device = device

        self.blocks = parse_cfg(self.network_name)
        self.layers = nn.ModuleList()
        in_channels, out_channels, out_channels_list = 3, 3, list()
        for idx, mdef in enumerate(self.blocks[1:]) :
            layer = nn.Sequential()
            if mdef['type'] == 'convolutional' :
                # conv_layer
                try :
                    has_bn = int(mdef['batch_normalize']) > 0
                except :
                    has_bn = False
                activation = mdef['activation']
                kernel_size, stride, need_pad, out_channels = int(mdef['size']), int(mdef['stride']), \
                                                                int(mdef['pad']), int(mdef['filters'])
                # adjust to different class number
                if self.blocks[idx + 2]['type'] == 'yolo' :
                    out_channels = 3 * (4 + 1 + self.class_num)
                padding = (kernel_size - 1) // 2 if need_pad else 0
                layer.add_module('conv_%d' % idx, nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                                            padding=padding, bias=not has_bn))

                # add BatchNorm
                if has_bn :
                    layer.add_module('bn_%d' % idx, nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5))

                # activation layer
                if activation == 'leaky' :
                    layer.add_module('leakyRelu_%d' % idx, nn.LeakyReLU(negative_slope=0.1, inplace=True))

            elif mdef['type'] == 'maxpool' :
                kernel_size, stride = int(mdef['size']), int(mdef['stride'])
                if stride == 1 :
                    # Refer to http://sofasofa.io/forum_main_post.php?postid=1003549
                    out_channels = in_channels
                    if kernel_size == 2 :
                        layer.add_module('zeropad_%d' % idx, nn.ZeroPad2d((0, 1, 0, 1)))
                else :
                    out_channels = in_channels
                layer.add_module('maxpool_%d' % idx, nn.MaxPool2d(kernel_size=kernel_size,
                                                                    stride=stride, padding=(kernel_size - 1) // 2))

            elif mdef['type'] == 'upsample' :
                stride = int(mdef['stride'])
                layer.add_module('upsample_%d' % idx, nn.Upsample(scale_factor=stride, mode='nearest'))
                out_channels = in_channels

            elif mdef['type'] == 'route' :
                mdef['layers'] = mdef['layers'].split(',')
                integrated_layers = [int(layer_id) for layer_id in mdef['layers']]

                integrated_layers = [layer_id - idx if layer_id > 0 else layer_id for layer_id in integrated_layers]
                mdef['layers'] = integrated_layers

                layer.add_module('route_%d' % idx, EmptyLayer())

                out_channels = 0
                for layer_id in integrated_layers :
                    out_channels += out_channels_list[layer_id + idx]

            elif mdef['type'] == 'shortcut' :
                bypass_layer_idx, activation = int(mdef['from']), mdef['activation']
                bypass_layer_idx = bypass_layer_idx - idx if bypass_layer_idx > 0 else bypass_layer_idx
                layer.add_module('shortcut_%d' % idx, EmptyLayer())
                if activation == 'leaky':
                    layer.add_module('leakyRelu_%d' % idx, nn.LeakyReLU(inplace=True))
                out_channels = out_channels_list[bypass_layer_idx + idx]

            elif mdef['type'] == 'yolo' :
                ignore_thresh = float(mdef['ignore_thresh'])
                str_masks = mdef['mask'].split(',')
                masks = [int(mask) for mask in str_masks]

                str_anchors = mdef['anchors'].split('  ')
                str_anchors = [str_anchor.split(',')[:2] for str_anchor in str_anchors]
                anchors = [(int(anchor_couple[0]), int(anchor_couple[1])) for anchor_couple in str_anchors]

                current_anchors = np.array([anchors[i] for i in masks])
                layer.add_module('yolo_%d' % idx, YoloLayer(current_anchors))

                out_channels = in_channels

            self.layers.append(layer)
            in_channels = out_channels
            out_channels_list.append(out_channels)

        self.loss_function = YoloLossLayer(self.class_num, ignore_thresold, self.coord_scale, self.conf_scale, self.cls_scale)
        self.predictor = YoloPredictLayer(self.conf_thresold, self.nms_thresold)

    def load_weights(self, weight_file_path) :
        with open(weight_file_path, 'rb') as weight_file :
            # head is 1. Major version number  2. Minor Version Number 3. Subversion number
            # 4,5. Images seen by the network (during training)
            header = np.fromfile(weight_file, dtype=np.int32, count=5)
            # weight data
            weights = np.fromfile(weight_file, dtype=np.float32)
            ptr = 0
            cutoff = None
            if 'darknet53.conv.74' in weight_file_path :
                cutoff = 75
            elif 'yolov3-tiny.conv.15' in weight_file_path:
                cutoff = 15
            for idx, mdef in enumerate(self.blocks[1:]) :
                if idx == cutoff :
                    break

                if mdef['type'] != 'convolutional' :
                    continue

                try :
                    has_bn = int(mdef['batch_normalize']) > 0
                except :
                    has_bn = False

                conv = self.layers[idx][0]
                if has_bn :
                    bn = self.layers[idx][1]

                    # load the biases and weights for the Convolutional layers
                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).to(self.device)
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).to(self.device)
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).to(self.device)
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).to(self.device)
                    ptr += num_bn_biases
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else :
                    # load the biases for the Convolutional layers
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                # load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def forward(self, x, targets=None, train_infos=None) :
        batch_size, input_dim = x.size(0), x.size(2)
        hidden_outputs = dict()
        yolo_outputs = list()
        for idx, mdef in enumerate(self.blocks[1:]) :

            if mdef['type'] == 'convolutional' or mdef['type'] == 'upsample' or mdef['type'] == 'maxpool' :
                x = self.layers[idx](x)

            elif mdef['type'] == 'route' :
                integrated_layers = mdef['layers']
                if len(integrated_layers) == 1 :
                    x = hidden_outputs[idx + integrated_layers[0]]
                else :
                    x = torch.cat([hidden_outputs[idx + layer_id] for layer_id in integrated_layers], dim=1)

            elif mdef['type'] == 'shortcut' :
                bypass_layer_idx = int(mdef['from'])
                bypass_layer_idx = bypass_layer_idx - idx if bypass_layer_idx > 0 else bypass_layer_idx
                x = hidden_outputs[idx - 1] + hidden_outputs[idx + bypass_layer_idx]

            elif mdef['type'] == 'yolo' :
                x = self.layers[idx][0](x, input_dim)
                yolo_outputs.append(x)

            hidden_outputs[idx] = x

        predictions      = torch.cat([yolo_output[0] for yolo_output in yolo_outputs], dim=1)
        priori_boxes     = torch.cat([yolo_output[1] for yolo_output in yolo_outputs], dim=0)
        featuremap_sizes = [yolo_output[2] for yolo_output in yolo_outputs]

        # train
        if self.training :
            loss = self.loss_function(predictions, priori_boxes, featuremap_sizes, input_dim, targets, train_infos)
            return loss

        # eval
        prediction_per_sample = self.predictor(predictions)
        return prediction_per_sample