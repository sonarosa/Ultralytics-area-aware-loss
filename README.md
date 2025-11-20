# Ultralytics Area-Aware Loss for YOLO  

## Overview  
This repository provides a suite of **area-aware / scale-aware loss functions** designed for object detection, with a focus on improving performance on small objects and handling class/scale imbalance. These custom losses are located in the `loss_functions/` folder. The `ultralytics/` folder contains or links to the modified version of the Ultralytics YOLO codebase (or integration code) that allows you to plug in one of these custom losses into a YOLO training pipeline.

## Proposed Loss – “Area-Aware Loss”  
### Motivation  
- Small objects often contribute less to the overall loss in standard detection training, so models tend to under-perform on them.  
- Traditional losses (e.g., BCE for classification, IoU for bounding boxes) treat all objects the same regardless of their size or area in the image.  
- The area-aware losses in this repo modulate the loss contributions according to the object area (or scale): smaller objects get **greater weighting** (or alternative scheduling) so their signal is amplified relative to bigger objects.

### Key Characteristics  
- Each ground-truth detection instance is associated with an *area factor* (for example: small objects have a higher factor, large objects a lower factor).  
- The classification loss term is multiplied (or otherwise modulated) by the object’s area factor → making smaller objects more influential in gradient updates.  
- Many variants optionally include smoothing, bounding thresholds (so extremely tiny objects don’t dominate), or additional regularization to keep training stable.  
- The objective: **“Make the model pay attention to the small / under-represented ones as much as the large ones.”**  
- Empirical experiments (see paper) demonstrate improved mAP on small-object subsets and better recall for small-scale classes under heavy imbalance.

### Usage in This Repo  
- Several variants of area-aware loss are provided in `loss_functions/`, for example: `area_aware_bce.py`, `smoothed_baa_bce.py`, etc.  
- Each file exports either:  
  - a function `compute_loss(preds, targets, …)` returning a scalar loss tensor  
  - or a class `CustomLoss(...)` with `forward(preds, targets)` method  
- Your modified YOLO training pipeline (in `ultralytics/`) is set up so you can swap in any of these custom losses for the classification head (and optionally objectness if applicable).

---

## How to Run a Custom Loss  
### Option A: Via modified Ultralytics CLI (if supported)  
If your `ultralytics/` codebase supports a `--loss_module` or similar flag:

```bash
python -m ultralytics.train \
  --data data/your_data.yaml \
  --weights yolov8n.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --loss_module loss_functions.smoothed_bce_loss
