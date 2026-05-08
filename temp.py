import os
import json
import time
from ultralytics import RTDETR
from small_utils import get_cost_time


def main():
    # 所有的训练集图片路径
    imgs_dir = "/data/mml/data_debugging_data/datasets/VisDrone-yolo/origin/train/images"
    model = RTDETR(f"runs/detect/train/weights/epoch0.pt")
    results = model(imgs_dir, stream=True)
    print()

if __name__ == "__main__":
    main()

