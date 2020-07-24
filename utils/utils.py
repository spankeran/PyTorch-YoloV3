#-*-coding:utf-8-*-
"""
some functions for train or test
"""
import os
import os.path as osp
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def time2str() :
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

def checkdir(path) :
    """
    may make directory
    :param path: path of directory
    :return: path
    """
    if not osp.exists(path) :
        os.makedirs(path)
    return path

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        try :
            nn.init.constant_(m.bias.data, 0.)
        except :
            pass

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1., 0.02)
        nn.init.constant_(m.bias.data, 0.)

def get_latest_file(dir_path) :
    """get the file path which is the newly creat"""
    all_files = os.listdir(dir_path)
    if all_files :
        sorted_files = sorted(all_files, key=lambda x : osp.getctime(osp.join(dir_path, x)))
        return sorted_files[-1]
    return

def warm_up_lr(initial_lr, iter, optimizer, burnin_number) :
    lr = initial_lr * (iter * 1.0 / burnin_number) ** 4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_train_ckpt(modules_optims, epoch, train_infos, ckpt_file) :
    """
    Save state_dict's of modules/optimizers to file during training
    :param modules_optims: A list, which members are either torch.nn.optimizer
    or torch.nn.Module.
    :param ep: the current epoch number
    :param ckpt_file: The file path.
    """
    state_dicts = [m.state_dict() for m in modules_optims]
    ckpt = dict(state_dicts=state_dicts, epoch=epoch, train_infos=train_infos)
    checkdir(os.path.dirname(os.path.abspath(ckpt_file)))
    torch.save(ckpt, ckpt_file)

def load_train_ckpt(modules_optims, ckpt_file, device):
    """
    Load state_dict's of modules/optimizers from file.
    :param modules_optims: A list, which members are either torch.nn.optimizer
      or torch.nn.Module.
    :param ckpt_file: The file path.
    :param load_to_cpu: Boolean. Whether to transform tensors in modules/optimizers
      to cpu type.
    """
    ckpt = torch.load(ckpt_file, map_location=device)
    for m, sd in zip(modules_optims, ckpt['state_dicts']):
        if m:
            m.load_state_dict(sd)
    return ckpt['epoch'], ckpt['train_infos']