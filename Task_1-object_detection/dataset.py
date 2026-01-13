import os
import json
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

class SyntheticDataset(Dataset):
    def __init__(self, root_dir, num_samples=500, img_size=224, mode='train'):
        self.root_dir = root_dir
        self.img_size = img_size
        self.mode = mode
        self.classes = ['circle', 'rectangle', 'triangle', 'star', 'pentagon']
        self.num_classes = len(self.classes)

        self.images_dir = os.path.join(root_dir, mode, 'images')
        self.labels_dir = os.path.join(root_dir, mode, 'labels')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        self.samples = []
        existing = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]

        if len(existing) < num_samples:
            print(f"generating {num_samples} {mode} samples...")
            self.generate_dataset(num_samples)

        for f in os.listdir(self.images_dir):
            if f.endswith('.png'):
                img_path = os.path.join(self.images_dir, f)
                lbl_path = os.path.join(self.labels_dir, f.replace('.png', '.json'))
                if os.path.exists(lbl_path):
                    self.samples.append((img_path, lbl_path))

    def generate_dataset(self, num_samples):
        for i in range(num_samples):
            img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 240

            num_objects = random.randint(1, 3)
            objects = []

            for _ in range(num_objects):
                cls_idx = random.randint(0, self.num_classes - 1)
                cls_name = self.classes[cls_idx]

                size = random.randint(25, 50)
                x = random.randint(size, self.img_size - size)
                y = random.randint(size, self.img_size - size)

                color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))

                if cls_name == 'circle':
                    cv2.circle(img, (x, y), size//2, color, -1)
                    x1, y1 = x - size//2, y - size//2
                    x2, y2 = x + size//2, y + size//2

                elif cls_name == 'rectangle':
                    x1, y1 = x - size//2, y - size//3
                    x2, y2 = x + size//2, y + size//3
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

                elif cls_name == 'triangle':
                    pts = np.array([[x, y - size//2],
                                   [x - size//2, y + size//2],
                                   [x + size//2, y + size//2]], np.int32)
                    cv2.fillPoly(img, [pts], color)
                    x1, y1 = x - size//2, y - size//2
                    x2, y2 = x + size//2, y + size//2

                elif cls_name == 'star':
                    pts = []
                    for j in range(5):
                        angle = j * 72 - 90
                        px = int(x + size//2 * np.cos(np.radians(angle)))
                        py = int(y + size//2 * np.sin(np.radians(angle)))
                        pts.append([px, py])
                        angle2 = angle + 36
                        px2 = int(x + size//4 * np.cos(np.radians(angle2)))
                        py2 = int(y + size//4 * np.sin(np.radians(angle2)))
                        pts.append([px2, py2])
                    pts = np.array(pts, np.int32)
                    cv2.fillPoly(img, [pts], color)
                    x1, y1 = x - size//2, y - size//2
                    x2, y2 = x + size//2, y + size//2

                elif cls_name == 'pentagon':
                    pts = []
                    for j in range(5):
                        angle = j * 72 - 90
                        px = int(x + size//2 * np.cos(np.radians(angle)))
                        py = int(y + size//2 * np.sin(np.radians(angle)))
                        pts.append([px, py])
                    pts = np.array(pts, np.int32)
                    cv2.fillPoly(img, [pts], color)
                    x1, y1 = x - size//2, y - size//2
                    x2, y2 = x + size//2, y + size//2

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(self.img_size, x2)
                y2 = min(self.img_size, y2)

                objects.append({
                    'class': cls_idx,
                    'class_name': cls_name,
                    'bbox': [x1, y1, x2, y2]
                })

            if self.mode == 'train':
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)
                    for obj in objects:
                        x1, y1, x2, y2 = obj['bbox']
                        obj['bbox'] = [self.img_size - x2, y1, self.img_size - x1, y2]

                brightness = random.randint(-30, 30)
                img = np.clip(img.astype(np.int32) + brightness, 0, 255).astype(np.uint8)

            img_path = os.path.join(self.images_dir, f'img_{i:04d}.png')
            lbl_path = os.path.join(self.labels_dir, f'img_{i:04d}.json')

            cv2.imwrite(img_path, img)
            with open(lbl_path, 'w') as f:
                json.dump(objects, f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        with open(lbl_path, 'r') as f:
            objects = json.load(f)

        boxes = []
        labels = []
        for obj in objects:
            boxes.append(obj['bbox'])
            labels.append(obj['class'])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        return img, target

def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets
