
from ultralytics import RTDETR
from small_utils import label_replace,read_yaml


def main():
    
    model = RTDETR("rtdetr-l.pt")
    train_labels_dir = f"{exp_data_root}/fault_inject/{inject_ratio}/{dataset_name}/yolo_format/labels_train"
    val_labels_dir = f"{exp_data_root}/fault_inject/{inject_ratio}/{dataset_name}/yolo_format/labels_val"
    label_replace(dataset_name,train_labels_dir,val_labels_dir)
    results = model.train(data=f"data/{dataset_name}.yaml",epochs=100,imgsz=640,batch=64,device=gpu_ids,
                        save_period = 1,
                        amp=False,
                        project=f"{exp_data_root}/models/{dataset_name.lower()}/rtdetr/fault_train/{inject_ratio}",
                        name=f"repeat_{repeat_id}",
                        exist_ok=True)

if __name__ == "__main__":
    config = read_yaml("config.yaml")
    exp_data_root = config["exp_data_dir"]
    inject_ratio = 0.1
    gpu_ids = [0]
    for dataset_name in ["voc","kitti","visdrone"]:
        for repeat_id in range(1,11):
            main()
