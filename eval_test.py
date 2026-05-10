from ultralytics import RTDETR
model = RTDETR("/data/mml/data_debugging_data/models/kitti_8/rtdetr/clean/weights/best.pt")
metrics = model.val()
print(metrics.box.map50)