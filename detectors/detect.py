import argparse
import bios
import imageio
import cv2

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
    p.add_argument("--video",
                    type=str,
                    help="input video path")
    p.add_argument("--tmp_folder",
                    type=str,
                    help="tmp folder path")
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

    for x in range(0, int(args.frame_cnt), int(args.segment_size)):
        logger.info(f"Performing detection on frame {x}")
        video = imageio.get_reader(args.video, "ffmpeg")
        image = video.get_data(x)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_path = f"{args.tmp_folder}/frame.jpeg"
        cv2.imwrite(image_path, image)
        
        detector.find_heads(img_path=image_path,
                            cfg=detector_cfg)
