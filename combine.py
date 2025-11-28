import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def make_masking_mosaic():
    """
    Combine six images into a 2x3 mosaic for Figure 2.1.
    Adjust the paths/order below to match the images you want.
    """

    # Paths in the order you want them to appear:
    # (a) (b) (c)
    # (d) (e) (f)
    img_paths = [
        PROJECT_ROOT / "masking_outputs" / "01_gray.png",  # (b)
        PROJECT_ROOT / "masking_outputs" / "02_thresh.png",# (c)
        PROJECT_ROOT / "masking_outputs" / "03_thresh_closed.png",  # (d)
        PROJECT_ROOT / "masking_outputs" / "04_edges_canny.png",    # (e)
        PROJECT_ROOT / "masking_outputs" / "05_mask.png",    # (e)
        PROJECT_ROOT / "masking_outputs" / "06_extracted_pcb.png",  # (f)
    ]

    images = []
    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        images.append(img)

    # Resize all images to the same size (pick what looks good in your report)
    target_w, target_h = 640, 360  # width, height
    images_resized = [cv2.resize(im, (target_w, target_h)) for im in images]

    # Build 2x3 grid: first row = (a,b,c), second row = (d,e,f)
    row1 = np.hstack(images_resized[0:3])
    row2 = np.hstack(images_resized[3:6])
    mosaic = np.vstack([row1, row2])

    out_dir = PROJECT_ROOT / "masking_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure_2_1_mosaic.png"

    cv2.imwrite(str(out_path), mosaic)
    print(f"Saved mosaic to: {out_path.resolve()}")

if __name__ == "__main__":
    make_masking_mosaic()
