
from ultralytics import RTDETR
from small_utils import label_replace

exp_data_root = "/data/mml/data_debugging_data/ProAFL_data"
model = RTDETR("rtdetr-l.pt")
dataset_name = "visdrone" # voc|visdrone|kitti
inject_ratio = 0.1
repeat_id = 1
train_labels_dir = f"{exp_data_root}/fault_inject/{inject_ratio}/{dataset_name}/yolo_format/labels_train"
val_labels_dir = f"{exp_data_root}/fault_inject/{inject_ratio}/{dataset_name}/yolo_format/labels_val"
label_replace(dataset_name,train_labels_dir,val_labels_dir)

results = model.train(data=f"data/{dataset_name}.yaml",epochs=100,imgsz=640,batch=64,device=[1],
                      save_period = 1,
                      amp=False,
                      project=f"{exp_data_root}/models/{dataset_name.lower()}/rtdetr/fault_train/{inject_ratio}",
                      name=f"repeat_{repeat_id}",
                      exist_ok=True)
# results = model.train(resume=True,project="/data/mml/data_debugging_data/models/visdrone/rtdetr",name="train")
