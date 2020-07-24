#-*-coding:utf-8-*-
"""
some functions for logging
"""
import logging.config
import os.path as osp
import random
import matplotlib.pyplot as plt
import prettytable as pt

from utils.utils import checkdir

BASE_DIR = osp.dirname(osp.abspath(__file__))

log_dir = osp.join(checkdir(osp.join(osp.dirname(BASE_DIR), 'results')), 'log')
info_log_path = osp.join(checkdir(log_dir), 'info.log')
error_log_path = osp.join(checkdir(log_dir), 'error.log')

LOGGING_DIC = {
    "version":1,
    "disable_existing_loggers": False,
    "formatters":{
        "simple":{
            "format":"[%(asctime)s]-[%(name)s]-[%(levelname)s]- %(message)s",
            "datefmt" : "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers":{
        "console":{
            "class":"logging.StreamHandler",
            "level":"DEBUG",
            "formatter":"simple",
            "stream":"ext://sys.stdout"
        },
        "info_file_handler":{
            "class":"logging.handlers.RotatingFileHandler",
            "level":"INFO",
            "formatter":"simple",
            "filename": info_log_path,
            "maxBytes":10485760,
            "backupCount":20,
            "encoding":"utf8"
        },
        "error_file_handler":{
            "class":"logging.handlers.RotatingFileHandler",
            "level":"ERROR",
            "formatter":"simple",
            "filename": error_log_path,
            "maxBytes":10485760,
            "backupCount":20,
            "encoding":"utf8"
        }
    },
    "loggers":{
        "YOLO":{
            "level":"INFO",
            "handlers":["console", "info_file_handler", "error_file_handler"],
            "propagate":True
        }
    },
}

class logger(object) :

    def __init__(self):
        logging.config.dictConfig(LOGGING_DIC)
        self.logger = logging.getLogger('YOLO')

    def info(self, infos) :
        self.logger.info(infos)

    def error(self, str) :
        self.logger.exception(str)

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / (self.count + 1e-20)

class train_infos(object) :
    __infos__ = {
        'giou_loss' : AverageMeter(),
        'conf_loss' : AverageMeter(),
        'cls_loss': AverageMeter(),
        'avg_iou' : AverageMeter(),
        'avg_conf' : AverageMeter(),
        'avg_cls_acc' : AverageMeter(),
        'precision' : AverageMeter(),
        'recall' : AverageMeter()
    }

    def __init__(self) :
        for k, v in self.__infos__.items() :
            self.__infos__[k].reset()

    def reset(self) :
        for k, v in self.__infos__.items() :
            self.__infos__[k].reset()

    def update(self, k, v) :
        self.__infos__[k].update(v)

    def get_str(self) :
        tb = pt.PrettyTable(field_names=list(self.__infos__.keys()))
        row = list()
        for k, v in self.__infos__.items() :
            row.append('{:.3f}'.format(v.avg))
        tb.add_row(row)
        return tb.get_string()

class train_results(object) :
    __infos__ = {
        'giou_loss' : list(),
        'conf_loss' : list(),
        'cls_loss': list(),
        'avg_iou': list(),
        'avg_conf': list(),
        'avg_cls_acc': list(),
        'precision': list(),
        'recall': list()
    }

    def __init__(self) :
        for k, v in self.__infos__.items() :
            self.__infos__[k].clear()

    def reset(self) :
        for k, v in self.__infos__.items() :
            self.__infos__[k].clear()

    def update(self, k, v) :
        self.__infos__[k].append(v)

    def draw(self, path) :
        plt.figure(figsize=(15, 10))
        plt.title("training infos")

        for i, (k,v) in enumerate(self.__infos__.items()):
            color = (random.random(), random.random(), random.random())
            plt.subplot(3, 3, i + 1)
            plt.xlabel("epochs")
            plt.ylabel(k)
            x = [j for j in range(len(v))]
            plt.plot(x, v, label="train", color=color, marker='.')
            plt.legend()

        plt.savefig(path, dpi=300, bbox_inches='tight')