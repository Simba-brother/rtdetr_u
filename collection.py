import os
import json
import time
from ultralytics import RTDETR
from small_utils import get_cost_time

def collect_one_epoch(model:RTDETR,imgs_dir):
    results = model.predict(imgs_dir, stream=True, batch=128, device="cuda:0",verbose=False)
    # 收集容器
    predicted_box_dict = {}
    predicted_box_id = 0
    for r in results:
        img_path = r.path
        img_basename = img_path.split("/")[-1]
        predicted_box_dict[img_basename] = {}
        predicted_bboxs = []
        xyxy_list = r.boxes.xyxy.tolist()
        clsid_list = r.boxes.cls.tolist()
        conf_list = r.boxes.conf.tolist()
        bbox = {}
        for xyxy,clsid,conf in zip(xyxy_list,clsid_list,conf_list):
            bbox["predicted_cls"] = int(clsid)
            bbox["bbox"] = xyxy
            bbox["conf"] = conf
            bbox["img_name"] = img_basename
            bbox["predicted_box_id"] = predicted_box_id
            predicted_box_id += 1
            predicted_bboxs.append(bbox)
        predicted_box_dict[img_basename]["predicted_bboxs"] = predicted_bboxs
    return predicted_box_dict

def main():
    # 所有的训练集图片路径
    imgs_dir = "/data/mml/data_debugging_data/datasets/VisDrone-yolo/origin/train/images"
    for epoch in range(Epochs):
        print(f"Epoch:{epoch}/{Epochs}...")
        e_start_timestamp = time.time()
        # 模型
        model = RTDETR(os.path.join(models_dir,f"epoch{epoch}.pt"))
        predicted_box_dict = collect_one_epoch(model,imgs_dir)
        print(f"推理图像数量:{len(predicted_box_dict.keys())}")
        # 保存收集容器
        save_dir = collect_p_box_dir
        os.makedirs(save_dir,exist_ok=True)
        save_json_file_name = f"epoch_{epoch}_predicted_bboxs.json"
        save_json_path = os.path.join(save_dir,save_json_file_name)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(predicted_box_dict, f, indent=4)
        print(f"数据保存在:{save_json_path}")
        e_end_timestamp = time.time()
        e_cost_timestamp = e_end_timestamp - e_start_timestamp
        e_cost_time = get_cost_time(e_cost_timestamp)
        print(f"该轮次耗时:{e_cost_time}")

if __name__ == "__main__":
    exp_data_root = "/data/mml/data_debugging_data"
    dataset_name = "VisDrone"
    model_name = "rtdetr"
    Epochs = 100
    collect_p_box_dir = os.path.join(exp_data_root,"collection_bbox_level",dataset_name,model_name,"predicted_bbox")
    models_dir = "/data/mml/data_debugging_data/models/visdrone/rtdetr/train/weights"
    main()