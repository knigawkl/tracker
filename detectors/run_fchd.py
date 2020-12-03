from __future__ import division

import os
import torch as t
import numpy as np
import time
import argparse
from PIL import Image

import fchd.src.utils as utils
import fchd.src.array_tool as at
from fchd.src.config import opt
from fchd.src.vis_tool import visdom_bbox
from fchd.src.config import opt
from fchd.src.head_detector_vgg16 import Head_Detector_VGG16
from fchd.trainer import Head_Detector_Trainer
from fchd.data.dataset import preprocess
from fchd.head_detection_demo import read_img

from utils.logger import logger
from base import BaseDetector

MODEL_PATH = "./fchd/checkpoints/sess:2/head_detector08120858_0.682282441835"


class FCHDDetector(BaseDetector):
    def find_heads(self, img_path: str, cfg: dict) -> []:
        file_id = utils.get_file_id(img_path)
        img, img_raw, scale = read_img(img_path)
        head_detector = Head_Detector_VGG16(ratios=[1], anchor_scales=[2,4])
        trainer = Head_Detector_Trainer(head_detector).cuda()
        trainer.load(MODEL_PATH)
        img = at.totensor(img)
        img = img[None, : ,: ,:]
        img = img.cuda().float()
        st = time.time()
        pred_bboxes_, _ = head_detector.predict(img, scale, mode='evaluate', thresh=THRESH)
        et = time.time()
        tt = et - st
        print ("[INFO] Head detection over. Time taken: {:.4f} s".format(tt))

        result = []
        for i in range(pred_bboxes_.shape[0]):
            ymin, xmin, ymax, xmax = pred_bboxes_[i,:]
            res = [xmin, ymin, xmax, ymax]
            result.append(res)
        return result
