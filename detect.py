"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import glob
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from engine.misc import dist_utils
from engine.core import YAMLConfig
from engine.solver import TASKS

def visualize(image, boxes, labels, scores, output_dir, image_name, class_names=None):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 获取原始图像的宽度和高度
    img_w, img_h = image.size

    for box, label, score in zip(boxes, labels, scores):
        # 提取归一化坐标
        x_center, y_center, w, h = box


        x_center = x_center * img_w
        y_center = y_center * img_h
        w = w * img_w
        h = h * img_h


        left = x_center - w / 2
        top = y_center - h / 2

        # 绘制矩形框，颜色改为蓝色
        rect = patches.Rectangle((left, top), w, h, linewidth=1, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)


        label_text = class_names[label] if class_names else str(label)

        # 修改文本颜色为黑色，背景颜色为黄色
        ax.text(left, top, f'{label_text}: {score:.2f}', color='black', fontsize=12,
                bbox=dict(facecolor='yellow', alpha=0.5))


    ax.axis('off')


    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight', pad_inches=0)
    plt.close()

def main(args):
    """
    主函数：加载模型、权重并对指定文件夹中的图像进行目标检测。
    """

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)


    cfg = YAMLConfig(args.config)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    model = solver.cfg.model


    checkpoint = torch.load(args.weights, map_location=args.device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()


    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),

    ])


    image_paths = glob.glob(os.path.join(args.input_dir, '*.jpg')) + \
                  glob.glob(os.path.join(args.input_dir, '*.png'))


    class_names = cfg.yaml_cfg.get('num_names', None)


    for image_path in image_paths:

        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(args.device)


        with torch.no_grad():
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_tensor)
            else:
                outputs = model(input_tensor)


        if isinstance(outputs, dict):

            boxes = outputs['pred_boxes'][0].cpu().numpy()
            logits = outputs['pred_logits'][0].cpu().numpy()
            scores = torch.sigmoid(torch.tensor(logits)).numpy().flatten()
            labels = np.zeros_like(scores, dtype=int)
        else:
            raise ValueError("输入格式有误")


        mask = scores > args.conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]


        image_name = os.path.basename(image_path)
        visualize(image, boxes, labels, scores, args.output_dir, image_name, class_names)


    dist_utils.cleanup()

if __name__ == '__main__':
    # 定义命令行参数
    parser = argparse.ArgumentParser(description='Validation script for object detection')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to the configuration file')
    parser.add_argument('-w', '--weights', type=str, required=True,
                        help='Path to the model weights')
    parser.add_argument('-i', '--input-dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Directory to save output images')
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='Confidence threshold for detection')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference (e.g., cuda, cpu)')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision for inference')
    parser.add_argument('--print-method', type=str, default='builtin',
                        help='Print method')
    parser.add_argument('--print-rank', type=int, default=0,
                        help='Print rank id')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    main(args)
