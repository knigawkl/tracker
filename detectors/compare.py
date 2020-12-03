import bios
import os
import time

from run_fchd import FCHDDetector
from run_yolo import YOLODetector
from run_ssd import SSDDetector

TEST_IMG_DIR = "test_imgs/"
IMAGE_PATHS = os.listdir(TEST_IMG_DIR)
IMAGE_PATHS = [TEST_IMG_DIR + path for path in IMAGE_PATHS]
cfg = bios.read("config.yaml")

if __name__ == "__main__":
    for img in IMAGE_PATHS:
        print(img)

        fchd = FCHDDetector()
        # yolo = YOLODetector()
        # ssd = SSDDetector()

        start = time.time()

        res = fchd.find_heads(img_path=img, cfg=cfg["fchd"])
        # res = yolo.find_heads(img_path=img, cfg=cfg["yolo"])
        # res = ssd.find_heads(img_path=img, cfg=cfg["ssd"])
        
        end = time.time()
        print("time", end - start)
        print("fchd", res)
