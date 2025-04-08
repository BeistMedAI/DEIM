DEIM for fracture detection

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

## 🧠 What is DEIM? | 什么是 DEIM？

DEIM（Dense One-to-One Matching + Matchability-Aware Loss）是一种提升 DETR 训练效率的方法。它通过：

- ✅ **Dense O2O 匹配**增加有效正样本数量
- ✅ MAL（匹配感知损失）提高低质量匹配样本的利用率
- ✅ 训练更快，收敛更快，精度更高

> Citation: Huang et al., CVPR 2025  
> [📄 Paper 链接](https://www.shihuahuang.cn/DEIM/) | [🔗 官方代码](https://github.com/ShihuaHuang95/DEIM)

---

## 📊 Results | 项目进展（持续更新）

我们将测试以下公开或私有数据集：

- ✅ PMF私有数据集
- ✅ FracAtlas公开数据集
---

## 🔬 Future Work | 后续改进方向

- [ ] 更适合医学图像的 MAL 优化
- [ ] 尝试 3D 模型（CT / MRI）
- [ ] 支持轻量部署（ONNX / TensorRT）
- [ ] 引入不确定性估计和伪标签辅助训练

---

