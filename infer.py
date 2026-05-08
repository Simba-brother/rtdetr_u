
from ultralytics import RTDETR

model = RTDETR("results/clean_visDrone_train/weights/best.pt")

img_path = "test_images/visdrone/0000006_00159_d_0000001.jpg"
img_paths = []
img_paths.append(img_path)
results = model.predict([img_path])
result = results[0]

# for result in results:
#     print(result.boxes.conf)          # 每个框的最大类别概率
#     print(result.boxes.cls)           # 最大概率对应类别
#     print(result.all_cls_scores)      # 每个保留框对所有类别的概率

# 第 i 个框
i = 3
sum_prob = 0.0
for cls_id, prob in enumerate(result.all_cls_scores[i]):
    print(cls_id, result.names[cls_id], float(prob))
    sum_prob += float(prob)
print(f"sum_prob:{sum_prob}")


'''
修改1
/home/mml/anaconda3/envs/ultralytics/lib/python3.10/site-packages/ultralytics/nn/modules/head.py
class RTDETRDecoder(nn.Module):
    def postprocess(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        boxes:  [batch_size, num_queries, 4]
        scores: [batch_size, num_queries, nc]

        return:
            [cx, cy, w, h, cls_prob_0, cls_prob_1, ..., cls_prob_nc-1, max_class_prob, class_index]
        """

        # scores_flat: [bs, num_queries * nc]
        scores_flat, index = scores.flatten(1).topk(self.num_queries)

        # 每个 top-k 结果对应哪个 query
        query_idx = index // self.nc

        # 每个 top-k 结果对应哪个类别
        cls_idx = index % self.nc

        # 取出对应 query 的 box
        selected_boxes = boxes.gather(
            dim=1,
            index=query_idx.unsqueeze(-1).expand(-1, -1, 4)
        )

        # 关键：取出对应 query 的所有类别概率
        selected_all_scores = scores.gather(
            dim=1,
            index=query_idx.unsqueeze(-1).expand(-1, -1, self.nc)
        )

        return torch.cat(
            [
                selected_boxes,                  # [bs, num_queries, 4]
                selected_all_scores,             # [bs, num_queries, nc]
                scores_flat[..., None],          # [bs, num_queries, 1]
                cls_idx[..., None].float(),      # [bs, num_queries, 1]
            ],
            dim=-1
        )

修改2
/home/mml/anaconda3/envs/ultralytics/lib/python3.10/site-packages/ultralytics/models/rtdetr/predict.py
class RTDETRPredictor(BasePredictor):
def postprocess(self, preds, img, orig_imgs):
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        print("preds.shape =", preds.shape)
        nc = len(self.model.names)

        bboxes = preds[..., :4]
        all_scores = preds[..., 4:4 + nc]
        scores = preds[..., 4 + nc:4 + nc + 1]
        labels = preds[..., 4 + nc + 1:4 + nc + 2]

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for bbox, score, label, score_all, orig_img, img_path in zip(
            bboxes, scores, labels, all_scores, orig_imgs, self.batch[0]
        ):
            bbox = ops.xywh2xyxy(bbox)

            # score: [300, 1] -> idx: [300]
            idx = score.squeeze(-1) > self.args.conf

            if self.args.classes is not None:
                classes = torch.tensor(self.args.classes, device=label.device)
                idx = (label == classes).any(1) & idx

            bbox = bbox[idx]
            score = score[idx]
            label = label[idx]
            score_all = score_all[idx]

            # 再截断 max_det
            bbox = bbox[: self.args.max_det]
            score = score[: self.args.max_det]
            label = label[: self.args.max_det]
            score_all = score_all[: self.args.max_det]

            pred = torch.cat([bbox, score, label], dim=-1)

            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh

            result = Results(orig_img, path=img_path, names=self.model.names, boxes=pred)

            # 每个保留框的所有类别概率
            result.all_cls_scores = score_all

            results.append(result)
        return results
'''

