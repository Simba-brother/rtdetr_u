from ultralytics import RTDETR
# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("/data/mml/data_debugging_data/models/kitti_8/rtdetr/error/weights/best.pt") # load a pretrained model (recommended for training)
metrics = model.val()
print(metrics.box.map50)