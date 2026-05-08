from ultralytics import RTDETR
# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt") # load a pretrained model (recommended for training)
# Display model information (optional)
# model.info()
results = model.train(data="data/VisDrone.yaml", epochs=100, imgsz=640, device=[0, 1], 
                      save=True, save_period=1)
