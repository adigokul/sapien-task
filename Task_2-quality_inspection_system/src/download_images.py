import urllib.request
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "images")

urls = {
    "non_defective": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/PCB_Spectrum.jpg/640px-PCB_Spectrum.jpg", "pcb_good_1.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Pcb_voorkant.jpg/640px-Pcb_voorkant.jpg", "pcb_good_2.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Pcb_image.jpg/640px-Pcb_image.jpg", "pcb_good_3.jpg"),
    ],
    "defective": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Damaged_PCB.jpg/640px-Damaged_PCB.jpg", "pcb_damaged_1.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Corrosion_on_PCB.jpg/640px-Corrosion_on_PCB.jpg", "pcb_corrosion_1.jpg"),
    ]
}

def download_file(url, save_path):
    try:
        print(f"downloading: {url}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            with open(save_path, 'wb') as f:
                f.write(data)
        print(f"saved: {save_path}")
        return True
    except Exception as e:
        print(f"failed: {e}")
        return False

def main():
    defective_dir = os.path.join(IMAGES_DIR, "defective")
    non_defective_dir = os.path.join(IMAGES_DIR, "non_defective")

    os.makedirs(defective_dir, exist_ok=True)
    os.makedirs(non_defective_dir, exist_ok=True)

    print("downloading pcb images...\n")

    success_count = 0
    for url, filename in urls["non_defective"]:
        path = os.path.join(non_defective_dir, filename)
        if download_file(url, path):
            success_count += 1

    for url, filename in urls["defective"]:
        path = os.path.join(defective_dir, filename)
        if download_file(url, path):
            success_count += 1

    print(f"\ndownloaded {success_count} images")

    if success_count < 3:
        print("\nnot enough images downloaded, will generate synthetic ones instead")

if __name__ == "__main__":
    main()
