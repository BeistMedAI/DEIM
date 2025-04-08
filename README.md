# 🦴 Fracture Detection with DEIM-DETR  
🩻 **基于 DEIM-DETR 的骨折线检测项目**

本项目基于 [CVPR 2025] 论文 [《DEIM: DETR with Improved Matching for Fast Convergence》](https://www.shihuahuang.cn/DEIM/)，旨在将 DEIM 框架应用并改进于医疗图像中的骨折线检测任务，例如 X 光或 CT 扫描图像。

> This project adapts the DEIM framework to the **medical domain**, focusing on automatic **fracture line detection** using Transformer-based object detection models.

---

## 🎯 Project Goals | 项目目标

- 🔬 将 DEIM 模型应用于骨折检测任务  
- 🧠 探索适用于医学图像的新型训练方法  
- 📈 对比基础模型与改进方法在医学数据集上的效果  
- 🧪 计划开发轻量化、可部署的检测系统

---

## 📁 Directory Structure | 项目结构

```bash
fracture-detection-deim/
├── configs/            # 配置文件（Config files）
├── datasets/           # 数据集加载器（Dataset loading and preparation）
├── models/             # 模型模块，包括DEIM结构（Model components: DEIM, MAL...）
├── tools/              # 训练、评估、可视化脚本（Train, evaluate, visualize）
├── visualize/          # 结果可视化（Visual output）
├── main.py             # 主运行文件（Main script）
└── README.md
```

---

## 🚀 Quick Start | 快速开始

### 1️⃣ Clone 仓库 | Clone the Repository

```bash
git clone https://github.com/your-username/fracture-detection-deim.git
cd fracture-detection-deim
```

### 2️⃣ 安装环境 | Install Environment

```bash
pip install -r requirements.txt
# 或使用 conda
conda create -n deim-med python=3.9
conda activate deim-med
```

### 3️⃣ 准备数据集 | Prepare Dataset

支持 COCO 格式的医疗图像数据集，例如：

```
datasets/
└── fracture_xray/
    ├── annotations/
    └── images/
```

### 4️⃣ 开始训练 | Start Training

```bash
python main.py --config configs/deim_fracture.yaml
```

---

## 🧠 What is DEIM? | 什么是 DEIM？

DEIM（Dense One-to-One Matching + Matchability-Aware Loss）是一种提升 DETR 训练效率的方法。它通过：

- ✅ **Dense O2O 匹配**增加有效正样本数量
- ✅ **MAL（匹配感知损失）**提高低质量匹配样本的利用率
- ✅ 训练更快，收敛更快，精度更高

> Citation: Huang et al., CVPR 2025  
> [📄 Paper 链接](https://www.shihuahuang.cn/DEIM/) | [🔗 官方代码](https://github.com/IDEA-Research/DEIM)

---

## 📷 Example Output | 示例输出

> ⚠️ 可视化示例图正在准备中，如果你已有对比图，我可以帮你添加！

---

## 📊 Results | 项目进展（持续更新）

我们将测试以下公开或私有数据集：

- ✅ MURA
- ✅ DeepFracture（或自建骨折数据）
- ⏳ 医院匿名化临床数据集（如有）

---

## 🔬 Future Work | 后续改进方向

- [ ] 更适合医学图像的 MAL 优化
- [ ] 尝试 3D 模型（CT / MRI）
- [ ] 支持轻量部署（ONNX / TensorRT）
- [ ] 引入不确定性估计和伪标签辅助训练

---

## 📌 Citation | 引用方式

```bibtex
@inproceedings{huang2025deim,
  title={DEIM: DETR with Improved Matching for Fast Convergence},
  author={Huang, Shihua and Lu, Zhichao and Cun, Xiaodong and Yu, Yongjun and Zhou, Xiao and Shen, Xi},
  booktitle={CVPR},
  year={2025}
}
```

---

## 📬 Contact | 联系方式

Maintainer | 项目维护者: Your Name  
Email | 邮箱: your.email@example.com  
GitHub: [@your-username](https://github.com/your-username)

---

## 🛡 License | 许可证

本项目使用 MIT 协议开源。  
Licensed under the MIT License.
