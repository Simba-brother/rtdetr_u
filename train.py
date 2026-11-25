
import os
import time
from ultralytics import RTDETR
from small_utils import label_replace,read_yaml
from small_utils import get_cost_time


def main():    
    model = RTDETR("rtdetr-l.pt")
    train_labels_dir = f"{exp_data_root}/fault_inject/{inject_ratio}/{dataset_name}/yolo_format/labels_train"
    val_labels_dir = f"{exp_data_root}/fault_inject/{inject_ratio}/{dataset_name}/yolo_format/labels_val"
    label_replace(dataset_name,train_labels_dir,val_labels_dir)
    results = model.train(data=f"data/{dataset_name}.yaml",epochs=50,imgsz=640,batch=32,device=gpu_ids,workers=8,
                        save_period = 1,
                        amp=False,
                        project=f"{exp_data_root}/models/{dataset_name.lower()}/rtdetr/fault_train/{inject_ratio}",
                        name=f"repeat_{repeat_id}",
                        exist_ok=True)

if __name__ == "__main__":
    print("PID:",os.getpid())
    config = read_yaml("config.yaml")
    exp_data_root = config["exp_data_dir"]
    inject_ratio = 0.1
    gpu_ids = [1]
    for dataset_name in ["visdrone"]: # ["voc","kitti","visdrone"]:
        dataset_start_time = time.time()
        repeat_ids = list(range(6,11))
        for repeat_id in repeat_ids:
            repeat_start_time = time.time()
            main()
            print(f"datasetname:{dataset_name}|repeat:{repeat_id}|costTime:{get_cost_time(time.time()-repeat_start_time)}") 
        avg_timestamp = (time.time()-dataset_start_time) / len(repeat_ids)
        print(f"datasetname:{dataset_name}|重复{len(repeat_ids)}次平均耗时:{get_cost_time(avg_timestamp)}")
            
