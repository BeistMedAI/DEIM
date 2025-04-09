import os
import torch
import cv2
import json
import argparse
from tqdm import tqdm
import numpy as np
from torchvision.ops import nms
from engine.core import YAMLConfig
from engine.solver import TASKS

class ObjectDetector:
    def __init__(self, model_path, config_path, val_json_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        self.model = self.load_model(model_path, config_path)
        self.classes = self.get_model_classes()
        self.file_name_to_id, self.image_sizes = self.load_image_info(val_json_path)

    @staticmethod
    def load_model(model_path, config_path):
        cfg = YAMLConfig(config_path)
        solver = TASKS[cfg.yaml_cfg['task']](cfg)
        model = solver.cfg.model
        checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.eval()
        return model

    def get_model_classes(self):
        return ['wear']  # 与val.json中的类别匹配

    def load_image_info(self, val_json_path):
        with open(val_json_path, 'r') as f:
            val_data = json.load(f)
        file_name_to_id = {img['file_name']: img['id'] for img in val_data['images']}
        image_sizes = {img['id']: (img['height'], img['width']) for img in val_data['images']}
        return file_name_to_id, image_sizes

    def preprocess_image(self, img_path, img_size=640):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        orig_h, orig_w = img.shape[:2]
        ratio = img_size / max(orig_h, orig_w)
        new_h = int(orig_h * ratio)
        new_w = int(orig_w * ratio)
        if new_h != orig_h or new_w != orig_w:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_h = img_size - new_h
        pad_w = img_size - new_w
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left
        img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        return img_tensor, (orig_h, orig_w), (ratio, (left, top), (new_w, new_h))

    def postprocess_boxes(self, detections, orig_size, ratio_pad):
        orig_h, orig_w = orig_size
        ratio, (pad_left, pad_top), (new_w, new_h) = ratio_pad
        input_size = 640

        if detections.numel() == 0:
            return torch.empty((0, 6), device=detections.device)

        boxes = detections[:, :4] * input_size
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w/2;  y1 = cy - h/2
        x2 = cx + w/2;  y2 = cy + h/2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        # 去掉 pad
        boxes[:, [0,2]] -= pad_left
        boxes[:, [1,3]] -= pad_top

        # 缩放回原始尺寸
        # new_w = orig_w * ratio, 所以 orig_w/new_w = 1/ratio
        scale_x = orig_w / new_w
        scale_y = orig_h / new_h
        boxes[:, [0,2]] *= scale_x
        boxes[:, [1,3]] *= scale_y

        # 裁剪到图像边界
        boxes[:, 0].clamp_(0, orig_w)
        boxes[:, 1].clamp_(0, orig_h)
        boxes[:, 2].clamp_(0, orig_w)
        boxes[:, 3].clamp_(0, orig_h)

        # 过滤无效框
        valid = (boxes[:,2] > boxes[:,0]) & (boxes[:,3] > boxes[:,1])
        detections = detections[valid]
        boxes = boxes[valid]

        if boxes.numel() > 0:
            detections[:, :4] = boxes
            return detections
        else:
            return torch.empty((0, 6), device=detections.device)

    def detect(self, img_path, conf_thresh=0.5, iou_thresh=0.45):
        file_name = os.path.basename(img_path)
        image_id = self.file_name_to_id[file_name]
        orig_size = self.image_sizes[image_id]

        img_tensor, _, ratio_pad = self.preprocess_image(img_path)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            detections = torch.zeros((0,6), device=self.device)

            if isinstance(outputs, dict) and 'pred_logits' in outputs:
                boxes = outputs['pred_boxes'][0]   
                logits = outputs['pred_logits'][0]
                scores = torch.sigmoid(logits).squeeze(-1)
                mask = scores > conf_thresh
                if mask.any():
                    boxes = boxes[mask]
                    scores = scores[mask]
                    labels = torch.zeros_like(scores, dtype=torch.long)

                    det = torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1).float()], dim=1)
                    det = self.postprocess_boxes(det, orig_size, ratio_pad)
                    if det.numel() > 0:
                        keep = nms(det[:, :4], det[:, 4], iou_thresh)
                        detections = det[keep]

            if detections.numel() == 0:
                return np.zeros((0,6))

            return detections.cpu().numpy()

def generate_coco_predictions(model_path, config_path, images_dir, output_json, val_json_path, conf_thresh=0.25, iou_thresh=0.45):
    detector = ObjectDetector(model_path, config_path, val_json_path)
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    predictions = []

    for img_path in tqdm(image_files, desc="Running inference"):
        file_name = os.path.basename(img_path)
        if file_name not in detector.file_name_to_id:
            print(f"Warning: {file_name} not found in val.json, skipping")
            continue
        image_id = detector.file_name_to_id[file_name]

        try:
            detections = detector.detect(img_path, conf_thresh, iou_thresh)
            for det in detections:
                x1, y1, x2, y2, score, category_id = det
                width = x2 - x1
                height = y2 - y1
                if width <= 0 or height <= 0:
                    print(f"Skipping invalid box for image {image_id}: [{x1}, {y1}, {x2}, {y2}]")
                    continue
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(category_id) + 1,  # COCO格式从1开始
                    "bbox": [float(x1), float(y1), float(width), float(height)],
                    "score": float(score)
                })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    with open(output_json, 'w') as f:
        json.dump(predictions, f)

    print(f"Predictions saved to {output_json}")
    print(f"Generated {len(predictions)} predictions across {len(image_files)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate COCO format predictions")
    parser.add_argument('--model', type=str, default='/repo/cxw/DEIM/DEIM/output/dfine_resnet18_Rail/best_stg1.pth')
    parser.add_argument('--config', type=str, default='/repo/cxw/DEIM/DEIM/configs/deim_dfine/dfine_resnet18_rail.yml')
    parser.add_argument('--images', type=str, default='/repo/cxw/Rail-COCO/val/images')
    parser.add_argument('--output', type=str, default='predictions.json')
    parser.add_argument('--val_json', type=str, default='/repo/cxw/Rail-COCO/val/annotations/val.json')
    parser.add_argument('--conf-thresh', type=float, default=0.25)
    parser.add_argument('--iou-thresh', type=float, default=0.45)

    args = parser.parse_args()
    generate_coco_predictions(args.model, args.config, args.images, args.output, args.val_json, args.conf_thresh, args.iou_thresh)
