# Custom Object Detection

object detector trained from scratch (no pretrained weights)

## what it does

detects 5 types of shapes: circle, rectangle, triangle, star, pentagon

## quick start

```bash
pip install torch torchvision opencv-python imageio

# train
python train.py

# evaluate
python evaluate.py

# visualize
python visualize.py
```

## results

- mAP@0.5: 63.52%
- FPS: 802.8
- Model size: 27.60 MB

## files

- dataset.py - generates synthetic dataset
- model.py - the detector architecture
- train.py - training code
- evaluate.py - calculates mAP and FPS
- visualize.py - makes detection images and gif
- Report.md - detailed report

## sample output

check outputs/detections/ folder for detection images
check outputs/detection_results.gif for animated results
