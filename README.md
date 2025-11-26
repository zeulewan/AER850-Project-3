# AER850 Project 3 – PCB Detection with YOLOv11

This project performs:
1. **PCB background masking** using OpenCV.
2. **Object detection** on PCB components using **YOLOv11** (Ultralytics).
3. **Evaluation** of the trained model on three test images.

All main logic is in `project3.py`.

---

## 1. Requirements

- **Operating system**: Linux / Windows / macOS
- **Conda** (Miniconda or Anaconda)
- **Python**: 3.11 (used in this setup)
- **GPU**: CUDA-capable (e.g., RTX 3090) + NVIDIA drivers installed
- **CUDA-enabled PyTorch** (so YOLO can use the GPU)
- Project files:
  - `project3.py`
  - Dataset YAML (e.g., `data/pcb_dataset.yaml`)
  - Training images/labels referenced by the YAML
  - Motherboard image (e.g., `data/motherboard_image.JPEG`)
  - Evaluation images in `data/Evaluation/`

---

## 2. Conda Environment Setup

From a terminal / shell:

```bash
# 1) Create environment
conda create -n aer850_proj3 python=3.11 -y

# 2) Activate environment
conda activate aer850_proj3

# 3) Install core Python dependencies
pip install ultralytics opencv-python numpy matplotlib pillow
```

```nvidia-smi``` to check gpu stats