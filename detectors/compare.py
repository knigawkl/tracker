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


fchd = FCHDDetector()
yolo = YOLODetector()
ssd = SSDDetector()


if __name__ == "__main__":
    for img in IMAGE_PATHS[6:]:
        print(img)

        start = time.time()
        res = fchd.find_heads(img_path=img, cfg=cfg["fchd"])
        end = time.time()
        print("fchd time", end - start)

        start = time.time()
        res = yolo.find_heads(img_path=img, cfg=cfg["yolo"])
        end = time.time()
        print("yolo time", end - start)

        start = time.time()
        res = ssd.find_heads(img_path=img, cfg=cfg["ssd"])
        end = time.time()
        print("ssd time", end - start)
