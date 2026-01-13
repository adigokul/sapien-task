import os
import cv2
import torch
import numpy as np
from model import SimpleDetector

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255)
]
CLASSES = ['circle', 'rectangle', 'triangle', 'star', 'pentagon']

def detect_image(model, image_path, device, conf_thresh=0.3):
    img = cv2.imread(image_path)
    orig_img = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    img_tensor = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        boxes, scores, labels = model.decode_predictions(outputs, conf_thresh=conf_thresh)

    boxes = boxes[0].cpu().numpy()
    scores = scores[0].cpu().numpy()
    labels = labels[0].cpu().numpy()

    h, w = orig_img.shape[:2]
    scale_x = w / 224
    scale_y = h / 224

    for i in range(len(boxes)):
        x1 = int(boxes[i][0] * scale_x)
        y1 = int(boxes[i][1] * scale_y)
        x2 = int(boxes[i][2] * scale_x)
        y2 = int(boxes[i][3] * scale_y)

        label = int(labels[i])
        score = scores[i]
        color = COLORS[label % len(COLORS)]

        cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)

        text = f"{CLASSES[label]}: {score:.2f}"
        cv2.putText(orig_img, text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return orig_img, len(boxes)

def create_detection_samples(model, device, input_dir, output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(input_dir) if f.endswith('.png')][:num_samples]

    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        result_img, num_detections = detect_image(model, img_path, device)

        output_path = os.path.join(output_dir, f"detected_{img_name}")
        cv2.imwrite(output_path, result_img)
        print(f"processed {img_name}: {num_detections} detections")

def create_gif(input_dir, output_path, fps=2):
    try:
        import imageio
        images = sorted([f for f in os.listdir(input_dir) if f.startswith('detected_')])

        frames = []
        for img_name in images[:20]:
            img_path = os.path.join(input_dir, img_name)
            img = imageio.imread(img_path)
            frames.append(img)

        if len(frames) > 0:
            imageio.mimsave(output_path, frames, fps=fps)
            print(f"gif saved to {output_path}")
    except ImportError:
        print("imageio not installed, skipping gif creation")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

    model = SimpleDetector(num_classes=5, img_size=224)
    checkpoint = torch.load('./checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print("\ngenerating detection visualizations...")
    create_detection_samples(
        model, device,
        input_dir='./data/test/images',
        output_dir='./outputs/detections',
        num_samples=20
    )

    print("\ncreating gif...")
    create_gif('./outputs/detections', './outputs/detection_results.gif')

    print("\ndone!")

if __name__ == '__main__':
    main()
