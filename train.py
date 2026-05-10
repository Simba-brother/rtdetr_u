from ultralytics import RTDETR
# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt") # load a pretrained model (recommended for training)
# Display model information (optional)
# model.info()
dataset_name = "KITTI_8" # VisDrone|KITTI_8|VOC2012
label_mode = "clean" # error|clean
results = model.train(data=f"data/{dataset_name}.yaml",epochs=100,imgsz=640,batch=32,device=[0,1],
                      project=f"/data/mml/data_debugging_data/models/{dataset_name.lower()}/rtdetr",
                      name=label_mode)

# results = model.train(resume=True,project="/data/mml/data_debugging_data/models/visdrone/rtdetr",name="train")
