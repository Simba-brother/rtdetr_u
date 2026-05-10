from ultralytics import RTDETR
model = RTDETR("/data/mml/data_debugging_data/models/voc2012/rtdetr/error/weights/best.pt")
metrics = model.val()
print(metrics.box.map50)