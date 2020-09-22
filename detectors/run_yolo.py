import os

import cv2

from yolo.frontend import YOLO
from utils.logger import logger
from base import BaseDetector

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class YOLODetector(BaseDetector):
    def find_heads(self, img_path: str, cfg: dict) -> []:
        yolo = YOLO(backend=cfg['backend'],
                    input_size=cfg['input_size'],
                    labels=cfg['labels'],
                    max_box_per_image=cfg['max_box_per_image'],
                    anchors=cfg['anchors'],
                    backend_path=cfg['backend_path'])

        yolo.load_weights(cfg["weights"])
        image = cv2.imread(img_path)
        boxes = yolo.predict(image)
        logger.info(f"Boxes found: {len(boxes)}")
        logger.info(boxes)

        image_h, image_w, _ = image.shape

        result = []
        for box in boxes:
            if box.get_score() < cfg['confidence_threshold']:
                continue
            xmin = int(box.xmin * image_w)
            ymin = int(box.ymin * image_h)
            xmax = int(box.xmax * image_w)
            ymax = int(box.ymax * image_h)
            res = [xmin, ymin, xmax, ymax]
            result.append([res])
        return result
