import argparse
import bios
import imageio
import cv2
import csv

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
    p.add_argument("--frame_cnt",
                    type=int,
                    required=True)
    return p.parse_args()


def detections_to_csv(detections, frame_num):
    filepath = f"{args.tmp_folder}/csv/frame{frame_num}.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for d in detections:
            writer.writerow(d)


if __name__ == "__main__":
    args = load_args()
    if args.detector == "ssd":
        detector = SSDDetector()
    elif args.detector == "yolo":
        detector = YOLODetector()
    detector_cfg = bios.read(args.cfg)[args.detector]

    video = imageio.get_reader(args.video, "ffmpeg")
    for x in range(0, int(args.frame_cnt)):
        logger.info(f"Performing detection on frame {x+1}/{args.frame_cnt}")
        image = video.get_data(x)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_path = f"{args.tmp_folder}/img/frame{x}.jpeg"
        cv2.imwrite(image_path, image)
        
        detections = detector.find_heads(img_path=image_path,
                                         cfg=detector_cfg)
        detections_to_csv(detections=detections,
                          frame_num=x)
