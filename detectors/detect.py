import argparse
import bios

from run_ssd import SSDDetector
from run_yolo import YOLODetector
from utils.logger import logger


def load_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector",
                    type=str,
                    required=True,
                    choices=["ssd", "yolo"],
                    help="detector name")
    p.add_argument("--cfg",
                    type=str,
                    help="detectors cfg",
                    default="config.yaml")
    p.add_argument("--segment_size",
                    type=int,
                    required=True)
    p.add_argument("--frame_cnt",
                    type=int,
                    required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = load_args()
    if args.detector == "ssd":
        detector = SSDDetector()
    elif args.detector == "yolo":
        detector = YOLODetector()
    detector_cfg = bios.read(args.cfg)[args.detector]

    logger.info(int(args.frame_cnt))
    logger.info(int(args.segment_size))

    for x in range(0, int(args.frame_cnt), int(args.segment_size)):
        logger.info(f"Performing detection on frame {x}")
        # detector.find_heads()
