from pathlib import Path
from ultralytics import RTDETR

exp_data_root = Path("/data/mml/data_debugging_data/ProAFL_data")

dataset_name = "kitti" # voc|kitti|visdrone
inject_ratio = 0.1
repeat_id = 1
method_name = "loss" # ours|datactive|objectlab|deepgini|entropy|margin|loss

# 与 retrain.py 的重训练输出目录保持一致
model_path = exp_data_root / "retrained_models" / method_name / dataset_name / "rtdetr" / str(inject_ratio) / f"repeat_{repeat_id}" / "weights" / "best.pt"

model = RTDETR(model_path)
# 在对应数据集的测试集上评估
metrics = model.val(data=f"data/{dataset_name}.yaml", split="test", imgsz=640, device=1)
print(f"mAP@0.5: {metrics.box.map50}")
