from __future__ import division

import os
import torch as t
from fchd.src.config import opt
from fchd.src.head_detector_vgg16 import Head_Detector_VGG16
from fchd.trainer import Head_Detector_Trainer
from PIL import Image
import numpy as np
from fchd.data.dataset import preprocess
import matplotlib.pyplot as plt 
import fchd.src.array_tool as at
from fchd.src.vis_tool import visdom_bbox
import argparse
import fchd.src.utils as utils
from fchd.src.config import opt
import time

SAVE_FLAG = 0
THRESH = 0.01
IM_RESIZE = False

def read_img(path):
    f = Image.open(path)
    if IM_RESIZE:
        f = f.resize((640,480), Image.ANTIALIAS)

    f.convert('RGB')
    img_raw = np.asarray(f, dtype=np.uint8)
    img_raw_final = img_raw.copy()
    img = np.asarray(f, dtype=np.float32)
    _, H, W = img.shape
    img = img.transpose((2,0,1))
    _, H, W = img.shape
    img = preprocess(img)
    _, o_H, o_W = img.shape
    scale = o_H / H
    return img, img_raw_final, scale 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="test image path")
    parser.add_argument("--model_path", type=str, default='./checkpoints/sess:2/head_detector08120858_0.682282441835')
    args = parser.parse_args()
    detect(args.img_path, args.model_path)




