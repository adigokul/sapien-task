import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import SyntheticDataset, collate_fn
from model import SimpleDetector

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0

def calculate_ap(recalls, precisions):
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap

def evaluate_map(model, dataloader, device, num_classes=5, iou_thresh=0.5):
    model.eval()
    all_detections = {c: [] for c in range(num_classes)}
    all_ground_truths = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for img_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            pred_boxes, pred_scores, pred_labels = model.decode_predictions(outputs, conf_thresh=0.3)

            for b in range(len(targets)):
                gt_boxes = targets[b]['boxes'].numpy()
                gt_labels = targets[b]['labels'].numpy()

                for i in range(len(gt_boxes)):
                    all_ground_truths[gt_labels[i]].append({
                        'img_idx': img_idx * len(targets) + b,
                        'box': gt_boxes[i],
                        'matched': False
                    })

                p_boxes = pred_boxes[b].cpu().numpy()
                p_scores = pred_scores[b].cpu().numpy()
                p_labels = pred_labels[b].cpu().numpy()

                for i in range(len(p_boxes)):
                    all_detections[p_labels[i]].append({
                        'img_idx': img_idx * len(targets) + b,
                        'box': p_boxes[i],
                        'score': p_scores[i]
                    })

    aps = []
    for c in range(num_classes):
        detections = all_detections[c]
        ground_truths = all_ground_truths[c]

        if len(ground_truths) == 0:
            continue

        detections = sorted(detections, key=lambda x: x['score'], reverse=True)

        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))

        for d_idx, det in enumerate(detections):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt['img_idx'] != det['img_idx']:
                    continue
                if gt['matched']:
                    continue

                iou = calculate_iou(det['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                tp[d_idx] = 1
                ground_truths[best_gt_idx]['matched'] = True
            else:
                fp[d_idx] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = calculate_ap(recalls, precisions)
        aps.append(ap)

    mAP = np.mean(aps) if len(aps) > 0 else 0
    return mAP, aps

def measure_fps(model, device, img_size=224, num_iterations=100):
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    end_time = time.time()

    fps = num_iterations / (end_time - start_time)
    return fps

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

    model = SimpleDetector(num_classes=5, img_size=224)
    checkpoint = torch.load('./checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    test_dataset = SyntheticDataset('./data', num_samples=200, img_size=224, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    print("\nevaluating mAP...")
    mAP, class_aps = evaluate_map(model, test_loader, device)
    classes = ['circle', 'rectangle', 'triangle', 'star', 'pentagon']

    print(f"\nmAP@0.5: {mAP*100:.2f}%")
    print("\nper-class AP:")
    for i, ap in enumerate(class_aps):
        print(f"  {classes[i]}: {ap*100:.2f}%")

    print("\nmeasuring inference speed...")
    fps = measure_fps(model, device)
    print(f"FPS: {fps:.1f}")

    model_size = get_model_size(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nmodel size: {model_size:.2f} MB")
    print(f"total parameters: {total_params:,}")

    results = {
        'mAP': mAP,
        'class_aps': {classes[i]: class_aps[i] for i in range(len(class_aps))},
        'fps': fps,
        'model_size_mb': model_size,
        'total_params': total_params
    }

    with open('./outputs/evaluation_results.txt', 'w') as f:
        f.write("=== Evaluation Results ===\n\n")
        f.write(f"mAP@0.5: {mAP*100:.2f}%\n\n")
        f.write("Per-class AP:\n")
        for i, ap in enumerate(class_aps):
            f.write(f"  {classes[i]}: {ap*100:.2f}%\n")
        f.write(f"\nFPS: {fps:.1f}\n")
        f.write(f"Model Size: {model_size:.2f} MB\n")
        f.write(f"Total Parameters: {total_params:,}\n")

    print("\nresults saved to ./outputs/evaluation_results.txt")
    return results

if __name__ == '__main__':
    evaluate()
