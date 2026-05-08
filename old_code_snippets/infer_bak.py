'''
模型推理脚本：保留每个预测框对所有类别的 probs
'''

import torch
from ultralytics import RTDETR
from ultralytics.engine.results import Results
from ultralytics.models.rtdetr.predict import RTDETRPredictor
from ultralytics.utils import ops
from ultralytics.utils import nms
from ultralytics import RTDETR
import torch

model = RTDETR("results/clean_visDrone_train/weights/best.pt")
img = "test_images/visdrone/0000006_00159_d_0000001.jpg"

# 先正常 predict 一次，让 predictor 初始化
_ = model.predict(img, imgsz=640, verbose=False)

predictor = model.predictor
predictor.setup_source(img)
# 读取和预处理图片
# dataset = predictor.setup_source(img)
batch = next(iter(predictor.dataset))

paths, im0s, s = batch[:3]

# 预处理成 Tensor: [B, 3, H, W]
im = predictor.preprocess(im0s)

with torch.no_grad():
    preds = predictor.model(im)

# RTDETR 输出一般是 (y, x)
# y: 已经 topk 后的结果 [B, num_queries, 6]
# x: 原始中间结果，包括 dec_bboxes, dec_scores
if isinstance(preds, (tuple, list)):
    y, x = preds
else:
    raise RuntimeError("当前 RTDETR 输出不是 tuple/list，请打印 type(preds), preds.shape 检查")

dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = x

# 取最后一层 decoder 输出
boxes_xywh = dec_bboxes[-1]          # [B, num_queries, 4]
cls_probs = dec_scores[-1].sigmoid() # [B, num_queries, num_classes]

# 第 0 张图
boxes_xywh = boxes_xywh[0]
cls_probs = cls_probs[0]

print(boxes_xywh.shape)
print(cls_probs.shape)

# 第 i 个预测 query 对所有类别的概率
i = 0
for cls_id, p in enumerate(cls_probs[i]):
    print(cls_id, model.names[cls_id], float(p))