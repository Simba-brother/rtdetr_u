from ultralytics import RTDETR
# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("results/error_visDrone_train/weights/best.pt") # load a pretrained model (recommended for training)
metrics = model.val()
print(metrics.box.map50)