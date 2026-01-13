import os
import sys
import json
from defect_detector import DefectDetector

def process_folder(input_folder, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(input_folder, "results")

    os.makedirs(output_folder, exist_ok=True)

    detector = DefectDetector()

    results = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(image_extensions):
            continue

        image_path = os.path.join(input_folder, filename)
        print(f"\nprocessing: {filename}")

        result = detector.inspect(image_path)

        if result is None:
            print(f"  skipped - couldnt load")
            continue

        base = os.path.splitext(filename)[0]

        annotated_path = os.path.join(output_folder, f"{base}_annotated.png")
        detector.draw_results(image_path, result, annotated_path)

        json_path = os.path.join(output_folder, f"{base}_results.json")
        detector.save_json(result, json_path)

        summary = {
            "image": filename,
            "defects": result["defect_count"],
            "score": result["quality_score"],
            "verdict": result["recommendation"].split(" - ")[0]
        }
        results.append(summary)

        print(f"  defects: {result['defect_count']}, score: {result['quality_score']}")

    summary_path = os.path.join(output_folder, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\ndone! processed {len(results)} images")
    print(f"results saved to: {output_folder}")

    pass_count = sum(1 for r in results if r["verdict"] == "PASS")
    fail_count = len(results) - pass_count
    print(f"\nPASS: {pass_count}, FAIL: {fail_count}")

    return results


def main():
    if len(sys.argv) < 2:
        print("usage: python batch_processor.py <input_folder> [output_folder]")
        return

    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None

    process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
