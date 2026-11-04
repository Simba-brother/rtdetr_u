
from ultralytics import RTDETR
from small_utils import label_replace
from pathlib import Path

exp_data_root = Path("/data/mml/data_debugging_data/ProAFL_data")

dataset_name = "visdrone" # voc|visdrone|kitti
inject_ratio = 0.1
repeat_id = 1

method_name = "ours" # ours|datactive
retrain_epoch = 25
if dataset_name == "visdrone":
    batch_size = 32
else:
    batch_size = 64

resume_model_pt = exp_data_root / "models" / dataset_name.lower() / "rtdetr" / "fault_train" / str(inject_ratio) / f"repeat_{repeat_id}" / "weights" / "epoch49.pt"

if method_name == "datactive":
    repair_root = exp_data_root / "repair" / "datactive" / "voc" / f"{inject_ratio}_repeat" / f"repeat_{repeat_id}" / "yolo_format"
else:
    repair_root = exp_data_root / "repair" / method_name / dataset_name / "rtdetr" / f"{inject_ratio}_repeat" / f"repeat_{repeat_id}" / "yolo_format"

train_labels_dir = repair_root / "labels_train"
val_labels_dir = repair_root / "labels_val"
output_dir = exp_data_root / "retrained_models" / method_name / dataset_name / "rtdetr" / str(inject_ratio) / f"repeat_{repeat_id}"

# 从注入故障训练得到的检查点加载权重做重训练
model = RTDETR(resume_model_pt)

label_replace(dataset_name,train_labels_dir,val_labels_dir)


results = model.train(data=f"data/{dataset_name}.yaml",epochs=retrain_epoch,imgsz=640,batch=batch_size,device=[1],
                      save_period=-1,
                      amp=False,
                      project=str(output_dir.parent),
                      name=output_dir.name,
                      exist_ok=True)

# 仅保留在 val 上表现最好的 best.pt，删除 last.pt
if model.trainer.last.exists():
    model.trainer.last.unlink()
    print(f"已删除 {model.trainer.last}，仅保留 best.pt: {model.trainer.best}")
print(f"重训练模型保存在 {output_dir}")

