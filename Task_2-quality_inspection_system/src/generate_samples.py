import cv2
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "images")

def make_clean_pcb(width=600, height=400):
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = 45
    img[:, :, 1] = 90
    img[:, :, 2] = 45

    for x in range(60, width-60, 100):
        for y in range(60, height-60, 80):
            cv2.rectangle(img, (x, y), (x+40, y+25), (50, 100, 50), -1)

    for x in range(120, width-120, 180):
        for y in range(100, height-100, 120):
            cv2.circle(img, (x, y), 10, (160, 160, 160), -1)
            cv2.circle(img, (x, y), 5, (180, 180, 180), -1)

    for x in range(100, width-100, 250):
        cv2.rectangle(img, (x, 40), (x+80, 65), (30, 30, 30), -1)
        for px in range(x+8, x+72, 12):
            cv2.rectangle(img, (px, 65), (px+6, 75), (160, 160, 160), -1)

    return img

def make_base_pcb(width=600, height=400):
    img = np.ones((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = 45
    img[:, :, 1] = 90
    img[:, :, 2] = 45

    for x in range(60, width-60, 100):
        for y in range(60, height-60, 80):
            cv2.rectangle(img, (x, y), (x+40, y+25), (50, 100, 50), -1)

    for x in range(120, width-120, 180):
        for y in range(100, height-100, 120):
            cv2.circle(img, (x, y), 10, (160, 160, 160), -1)
            cv2.circle(img, (x, y), 5, (180, 180, 180), -1)

    for x in range(100, width-100, 250):
        cv2.rectangle(img, (x, 40), (x+80, 65), (30, 30, 30), -1)
        for px in range(x+8, x+72, 12):
            cv2.rectangle(img, (px, 65), (px+6, 75), (160, 160, 160), -1)

    return img

def add_scratch(img, count=3):
    h, w = img.shape[:2]
    for _ in range(count):
        x1 = np.random.randint(80, w-80)
        y1 = np.random.randint(80, h-80)
        length = np.random.randint(80, 180)
        angle = np.random.uniform(0, np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), (120, 120, 120), 3)
    return img

def add_missing_component(img, count=3):
    h, w = img.shape[:2]
    for _ in range(count):
        x = np.random.randint(120, w-180)
        y = np.random.randint(120, h-120)
        cw = np.random.randint(35, 55)
        ch = np.random.randint(20, 35)
        cv2.rectangle(img, (x, y), (x+cw, y+ch), (25, 35, 25), -1)
        cv2.rectangle(img, (x, y), (x+cw, y+ch), (80, 80, 80), 2)
    return img

def add_solder_bridge(img, count=3):
    h, w = img.shape[:2]
    for _ in range(count):
        x = np.random.randint(150, w-150)
        y = np.random.randint(120, h-120)
        bw = np.random.randint(60, 100)
        bh = np.random.randint(12, 20)
        cv2.ellipse(img, (x, y), (bw//2, bh//2), np.random.randint(0, 180),
                   0, 360, (220, 220, 220), -1)
    return img

def add_discoloration(img, count=2):
    h, w = img.shape[:2]
    for _ in range(count):
        x = np.random.randint(80, w-120)
        y = np.random.randint(80, h-120)
        dw = np.random.randint(50, 100)
        dh = np.random.randint(40, 80)
        overlay = img.copy()
        cv2.ellipse(overlay, (x+dw//2, y+dh//2), (dw//2, dh//2), 0, 0, 360,
                   (60, 140, 200), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    return img

def create_annotations(defects, img_name, w, h):
    return {
        "image": img_name,
        "width": w,
        "height": h,
        "defects": defects
    }

def main():
    defective_dir = os.path.join(IMAGES_DIR, "defective")
    non_defective_dir = os.path.join(IMAGES_DIR, "non_defective")
    annotated_dir = os.path.join(IMAGES_DIR, "annotated")

    os.makedirs(defective_dir, exist_ok=True)
    os.makedirs(non_defective_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    print("making sample images...")
    print("\n=== GOOD PCBs (should PASS) ===")

    for i in range(3):
        img = make_clean_pcb()
        path = os.path.join(non_defective_dir, f"good_pcb_{i+1}.png")
        cv2.imwrite(path, img)
        print(f"made: {path}")

        ann = create_annotations([], f"good_pcb_{i+1}.png", 600, 400)
        ann_path = os.path.join(annotated_dir, f"good_pcb_{i+1}_annotations.json")
        with open(ann_path, 'w') as f:
            json.dump(ann, f, indent=2)

    print("\n=== BAD PCBs (should REJECT) ===")

    img = make_base_pcb()
    img = add_scratch(img, 4)
    path = os.path.join(defective_dir, "pcb_scratches.png")
    cv2.imwrite(path, img)
    print(f"made: {path}")
    ann = create_annotations([
        {"type": "scratch", "severity": "high", "count": 4}
    ], "pcb_scratches.png", 600, 400)
    with open(os.path.join(annotated_dir, "pcb_scratches_annotations.json"), 'w') as f:
        json.dump(ann, f, indent=2)

    img = make_base_pcb()
    img = add_missing_component(img, 4)
    path = os.path.join(defective_dir, "pcb_missing_parts.png")
    cv2.imwrite(path, img)
    print(f"made: {path}")
    ann = create_annotations([
        {"type": "missing_component", "severity": "critical", "count": 4}
    ], "pcb_missing_parts.png", 600, 400)
    with open(os.path.join(annotated_dir, "pcb_missing_parts_annotations.json"), 'w') as f:
        json.dump(ann, f, indent=2)

    img = make_base_pcb()
    img = add_solder_bridge(img, 4)
    path = os.path.join(defective_dir, "pcb_solder_bridge.png")
    cv2.imwrite(path, img)
    print(f"made: {path}")
    ann = create_annotations([
        {"type": "solder_bridge", "severity": "critical", "count": 4}
    ], "pcb_solder_bridge.png", 600, 400)
    with open(os.path.join(annotated_dir, "pcb_solder_bridge_annotations.json"), 'w') as f:
        json.dump(ann, f, indent=2)

    img = make_base_pcb()
    img = add_discoloration(img, 3)
    path = os.path.join(defective_dir, "pcb_discoloration.png")
    cv2.imwrite(path, img)
    print(f"made: {path}")
    ann = create_annotations([
        {"type": "discoloration", "severity": "high", "count": 3}
    ], "pcb_discoloration.png", 600, 400)
    with open(os.path.join(annotated_dir, "pcb_discoloration_annotations.json"), 'w') as f:
        json.dump(ann, f, indent=2)

    img = make_base_pcb()
    img = add_scratch(img, 2)
    img = add_missing_component(img, 2)
    img = add_solder_bridge(img, 2)
    img = add_discoloration(img, 1)
    path = os.path.join(defective_dir, "pcb_multiple_defects.png")
    cv2.imwrite(path, img)
    print(f"made: {path}")
    ann = create_annotations([
        {"type": "scratch", "severity": "medium", "count": 2},
        {"type": "missing_component", "severity": "high", "count": 2},
        {"type": "solder_bridge", "severity": "critical", "count": 2},
        {"type": "discoloration", "severity": "low", "count": 1}
    ], "pcb_multiple_defects.png", 600, 400)
    with open(os.path.join(annotated_dir, "pcb_multiple_defects_annotations.json"), 'w') as f:
        json.dump(ann, f, indent=2)

    print("\ndone!")
    print(f"good pcbs: {non_defective_dir}")
    print(f"bad pcbs: {defective_dir}")


if __name__ == "__main__":
    main()
