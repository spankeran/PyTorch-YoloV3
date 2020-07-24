# YoloV3
a simple PyTorch implementation of YoloV3 

[![](https://img.shields.io/badge/Python-3.7-yellow)](https://www.python.org/)
[![](https://img.shields.io/badge/PyTorch-1.3.1-brightgreen)](https://github.com/pytorch/pytorch)
[![](https://img.shields.io/badge/Numpy-1.15.1-red)](https://github.com/numpy/numpy/)
[![](https://img.shields.io/badge/Cv2-4.1.2-blue)](https://github.com/opencv/opencv)
[![](https://img.shields.io/badge/CUDA-8.0-orange)](https://developer.nvidia.com/cuda-downloads)

## Usage

### 1.prepare dataset

The folder structure is as follows
```
├── data
│   ├── dataset # put your dataset here, and name it as follows: VOC2007_trainval
│   ├── ...    
│
├── results # log files
├── weights # model weights
├── checkpoints
├── main.py # training code
```

### 2.train

```bash
python main.py --model_name choose_your_model_name  # yon can see more arguments in train.py
```

## Resutls

![](./assert/1.png)

![](./assert/2.png)

![](./assert/train_info_2020-04-13-03-42-57.png)

## Reference

[YoloV3 by ultralytics](https://github.com/ultralytics/yolov3)