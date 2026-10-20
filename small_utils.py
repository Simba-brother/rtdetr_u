
import os
import shutil
exp_data_root = "/data/mml/data_debugging_data/ProAFL_data"

def get_cost_time(cost_timetamp)->str:
    hours = int(cost_timetamp // 3600)  # 计算小时数
    minutes = int((cost_timetamp % 3600) // 60)  # 计算分钟数
    seconds = cost_timetamp % 60  # 计算剩余的秒数
    return f"{hours:02d}:{minutes:02d}:{seconds:02.0f}"

def is_directory_exists(path):
    return os.path.exists(path) and os.path.isdir(path)

def label_replace(dataset_name,train_labels_dir,val_labels_dir):
    cur_train_labels_dir = f"{exp_data_root}/datasets/{dataset_name}-yolo/train/labels"
    cur_val_labels_dir = f"{exp_data_root}/datasets/{dataset_name}-yolo/val/labels"
    if is_directory_exists(cur_train_labels_dir):
        shutil.rmtree(cur_train_labels_dir)
        print("Train labels directory exists and is now removed")
    else:
        print("训练集中不存在标签目录，无需删除")
    if is_directory_exists(cur_val_labels_dir):
        shutil.rmtree(cur_val_labels_dir)
        print("Val labels directory exists and is now removed")
    else:
        print("验证集中不存在标签目录，无需删除")
    shutil.copytree(train_labels_dir, cur_train_labels_dir)
    shutil.copytree(val_labels_dir, cur_val_labels_dir)
    print("Training label directory replacement completed")
    print("Validation label directory replacement completed")
    for labels_dir in (cur_train_labels_dir, cur_val_labels_dir):
        cache_path = labels_dir + ".cache"
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"已删除标签缓存: {cache_path}")
