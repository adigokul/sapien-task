import cv2
import numpy as np
import json
import os
from datetime import datetime


class DefectDetector:
    def __init__(self):
        self.defect_count = 0
        self.scratch_threshold = 100
        self.min_line_length = 70
        self.component_area_min = 1000
        self.solder_thresh = 215
        self.min_defect_area = 400

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return gray, blur, edges

    def get_severity(self, area, length=0):
        score = area + (length * 10)
        if score < 500:
            return "low"
        elif score < 2000:
            return "medium"
        elif score < 5000:
            return "high"
        else:
            return "critical"

    def find_scratches(self, img, gray, edges):
        defects = []
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.scratch_threshold,
                                minLineLength=self.min_line_length, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                if length < self.min_line_length:
                    continue

                min_x = min(x1, x2)
                max_x = max(x1, x2)
                min_y = min(y1, y2)
                max_y = max(y1, y2)

                pad = 5
                x = max(0, min_x - pad)
                y = max(0, min_y - pad)
                w = max(10, max_x - min_x + 2*pad)
                h = max(10, max_y - min_y + 2*pad)

                center_x = x + w // 2
                center_y = y + h // 2
                area = w * h

                conf = min(0.95, 0.75 + (length / 200) * 0.2)
                sev = self.get_severity(area, length)

                self.defect_count += 1
                defects.append({
                    "id": self.defect_count,
                    "type": "scratch",
                    "confidence": round(conf, 3),
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "center_x": center_x,
                    "center_y": center_y,
                    "severity": sev,
                    "area": area,
                    "desc": f"scratch found, length {length:.1f}px"
                })

        return defects[:5]

    def find_missing_parts(self, img, gray):
        defects = []
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < self.component_area_min or area > 10000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w) / h if h > 0 else 0

            if 0.3 < ratio < 3.0:
                center_x = x + w // 2
                center_y = y + h // 2
                conf = min(0.92, 0.75 + (area / 5000) * 0.15)
                sev = self.get_severity(area)

                self.defect_count += 1
                defects.append({
                    "id": self.defect_count,
                    "type": "missing_component",
                    "confidence": round(conf, 3),
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "center_x": center_x,
                    "center_y": center_y,
                    "severity": sev,
                    "area": int(area),
                    "desc": f"missing component maybe, area {area}px"
                })

        return defects[:5]

    def find_solder_bridges(self, img, gray):
        defects = []
        _, binary = cv2.threshold(gray, self.solder_thresh, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < self.min_defect_area or area > 8000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1

            if ratio > 1.5:
                center_x = x + w // 2
                center_y = y + h // 2
                conf = min(0.88, 0.75 + (ratio / 10) * 0.1)
                sev = self.get_severity(area)

                self.defect_count += 1
                defects.append({
                    "id": self.defect_count,
                    "type": "solder_bridge",
                    "confidence": round(conf, 3),
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "center_x": center_x,
                    "center_y": center_y,
                    "severity": sev,
                    "area": int(area),
                    "desc": f"solder bridge maybe, ratio {ratio:.2f}"
                })

        return defects[:5]

    def find_discoloration(self, img):
        defects = []
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([10, 50, 50])
        upper = np.array([30, 255, 200])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < self.min_defect_area * 2:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w // 2
            center_y = y + h // 2
            conf = min(0.85, 0.75 + (area / 10000) * 0.1)
            sev = self.get_severity(area)

            self.defect_count += 1
            defects.append({
                "id": self.defect_count,
                "type": "discoloration",
                "confidence": round(conf, 3),
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "center_x": center_x,
                "center_y": center_y,
                "severity": sev,
                "area": int(area),
                "desc": "color looks wrong, maybe damage"
            })

        return defects[:5]

    def get_quality_score(self, defects, img_area):
        if not defects:
            return 100.0

        weights = {"low": 5, "medium": 15, "high": 30, "critical": 50}
        penalty = 0

        for d in defects:
            penalty += weights.get(d["severity"], 10) * d["confidence"]

        return max(0, round(100 - penalty, 2))

    def get_recommendation(self, score, defects):
        if score >= 95:
            return "PASS - looks good"
        elif score >= 80:
            return "REVIEW - some small issues found"
        elif score >= 60:
            return "REWORK - need to fix some stuff"
        else:
            return "REJECT - too many problems"

    def inspect(self, image_path):
        self.defect_count = 0

        img = cv2.imread(image_path)
        if img is None:
            print(f"cant load image: {image_path}")
            return None

        gray, blur, edges = self.preprocess(img)

        all_defects = []
        all_defects.extend(self.find_scratches(img, gray, edges))
        all_defects.extend(self.find_missing_parts(img, gray))
        all_defects.extend(self.find_solder_bridges(img, gray))
        all_defects.extend(self.find_discoloration(img))

        img_area = img.shape[0] * img.shape[1]
        score = self.get_quality_score(all_defects, img_area)
        rec = self.get_recommendation(score, all_defects)

        result = {
            "image": image_path,
            "time": datetime.now().isoformat(),
            "has_defects": len(all_defects) > 0,
            "defect_count": len(all_defects),
            "defects": all_defects,
            "quality_score": score,
            "recommendation": rec
        }

        return result

    def draw_results(self, image_path, result, save_path=None):
        img = cv2.imread(image_path)

        colors = {
            "scratch": (0, 0, 255),
            "missing_component": (0, 165, 255),
            "solder_bridge": (0, 255, 255),
            "discoloration": (255, 0, 255)
        }

        for d in result["defects"]:
            bbox = d["bbox"]
            color = colors.get(d["type"], (255, 255, 255))

            cv2.rectangle(img, (bbox["x"], bbox["y"]),
                         (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                         color, 2)

            cv2.circle(img, (d["center_x"], d["center_y"]), 5, color, -1)

            label = f"{d['type']}: {d['confidence']:.2f}"
            cv2.putText(img, label, (bbox["x"], bbox["y"] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        info = f"Defects: {result['defect_count']} | Score: {result['quality_score']}"
        cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if save_path:
            cv2.imwrite(save_path, img)

        return img

    def save_json(self, result, output_path):
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        cleaned = convert(result)
        with open(output_path, 'w') as f:
            json.dump(cleaned, f, indent=2)


def main():
    import sys

    if len(sys.argv) < 2:
        print("usage: python defect_detector.py <image_path> [--save]")
        return

    image_path = sys.argv[1]
    save_output = "--save" in sys.argv

    detector = DefectDetector()

    print(f"\nchecking image: {image_path}")
    print("-" * 40)

    result = detector.inspect(image_path)

    if result is None:
        return

    print(f"time: {result['time']}")
    print(f"has defects: {result['has_defects']}")
    print(f"total defects: {result['defect_count']}")
    print(f"quality score: {result['quality_score']}/100")
    print(f"verdict: {result['recommendation']}")

    if result["defects"]:
        print("\ndefects found:")
        print("-" * 40)
        for d in result["defects"]:
            print(f"\n  defect #{d['id']}:")
            print(f"    type: {d['type']}")
            print(f"    confidence: {d['confidence']:.1%}")
            print(f"    center (x, y): ({d['center_x']}, {d['center_y']})")
            print(f"    severity: {d['severity']}")
            print(f"    area: {d['area']} pixels")
            print(f"    note: {d['desc']}")

    if save_output:
        output_dir = os.path.dirname(image_path) or "."
        base = os.path.splitext(os.path.basename(image_path))[0]

        img_out = os.path.join(output_dir, f"{base}_annotated.png")
        detector.draw_results(image_path, result, img_out)
        print(f"\nsaved annotated image: {img_out}")

        json_out = os.path.join(output_dir, f"{base}_results.json")
        detector.save_json(result, json_out)
        print(f"saved json: {json_out}")


if __name__ == "__main__":
    main()
