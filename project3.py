"""
AER850 Project 3 – PCB Masking + YOLOv11 Training + Evaluation

- Step 1: Object masking of motherboard_image.JPEG using OpenCV
- Step 2: YOLOv11 nano training on PCB dataset using Ultralytics
- Step 3: Evaluate 3 images in Evaluation/ using model.predict()
"""

import os

import cv2
import numpy as np
from ultralytics import YOLO


from pathlib import Path

# =========================
# CONFIG
# =========================

# Root of this project (folder that contains project3.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Paths ---
# Motherboard image is in the same folder as project3.py
MOTHERBOARD_IMAGE_PATH = PROJECT_ROOT / "motherboard_image.JPEG"

# Where to save masking results (this folder will be created)
MASK_OUTPUT_DIR = PROJECT_ROOT / "masking_outputs"

# Dataset YAML 
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"

# YOLO runs directory (Ultralytics default under this project)
MODEL_WEIGHTS_DIR = PROJECT_ROOT / "runs" / "detect"

# Folder with the 3 images you want to run predictions on
EVAL_IMAGES_DIR = PROJECT_ROOT / "prediction imgs"

# --- YOLO Training Hyperparameters ---
RUN_NAME = "pcb_yolo11n"   # or any name you like
EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 1024
DEVICE = 0                 # RTX 3090 = GPU 0


# =========================
# UTILS
# =========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# =========================
# STEP 1: OBJECT MASKING
# =========================

def step1_mask_motherboard(
    input_path: Path = MOTHERBOARD_IMAGE_PATH,
    output_dir: Path = MASK_OUTPUT_DIR,
):
    """
    1. Load original image
    2. Convert to grayscale
    3. Apply Gaussian blur
    4. Use Otsu thresholding to create a binary image
    5. Use Canny edge detection for edge figure
    6. Find largest external contour (assumed PCB outline)
    7. Create mask from that contour
    8. Extract PCB using cv2.bitwise_and
    9. Save: edges, threshold, mask, extracted image
    """
    ensure_dir(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Motherboard image not found at {input_path}")

    # 1. Load BGR image
    orig_bgr = cv2.imread(str(input_path))
    if orig_bgr is None:
        raise ValueError(f"Failed to read image at {input_path}")

    # 2. Grayscale
    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Otsu thresholding
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Invert if background/PCB polarity is wrong (simple heuristic)
    # We assume PCB should be foreground (white) occupying more area
    white_ratio = np.mean(thresh == 255)
    if white_ratio < 0.5:
        thresh = cv2.bitwise_not(thresh)

    # Optional morphological closing to fill small gaps in PCB region
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Canny edge detection (for edge figure)
    edges = cv2.Canny(blurred, 100, 200)

    # 6. Find contours on the binary mask
    contours, _ = cv2.findContours(
        thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError("No contours found – adjust thresholding parameters.")

    # Largest contour by area is assumed to be the PCB outline
    largest_contour = max(contours, key=cv2.contourArea)

    # 7. Create a blank mask and draw the largest contour as a filled shape
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], contourIdx=-1, color=255, thickness=cv2.FILLED)

    # Optional refinement (e.g., slight erosion/dilation) if edge is noisy
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 8. Extract PCB
    extracted_bgr = cv2.bitwise_and(orig_bgr, orig_bgr, mask=mask)

    # 9. Save intermediate and final images for use in report
    cv2.imwrite(str(output_dir / "01_gray.png"), gray)
    cv2.imwrite(str(output_dir / "02_thresh_otsu.png"), thresh)
    cv2.imwrite(str(output_dir / "03_thresh_closed.png"), thresh_closed)
    cv2.imwrite(str(output_dir / "04_edges_canny.png"), edges)
    cv2.imwrite(str(output_dir / "05_mask.png"), mask)
    cv2.imwrite(str(output_dir / "06_extracted_pcb.png"), extracted_bgr)

    print(f"[STEP 1] Saved masking outputs to: {output_dir.resolve()}")


# =========================
# STEP 2: YOLOv11 TRAINING
# =========================

def step2_train_yolo(
    data_yaml: Path = DATA_YAML,
    run_name: str = RUN_NAME,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    device: int | str = DEVICE,
):
    """
    Train YOLOv11 nano (yolo11n.pt) on provided PCB dataset.

    Ultralytics automatically creates runs/detect/<run_name> with:
    - best.pt, last.pt (weights)
    - metrics, confusion matrix, PR, P-confidence curves, etc.
    """
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found at {data_yaml}")

    print("[STEP 2] Loading YOLOv11 nano model (yolo11n.pt)...")
    model = YOLO("yolo11n.pt")  # downloads if not present

    print("[STEP 2] Starting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        name=run_name,
        # Optional extras:
        # workers=8,
        # patience=50,  # early stopping
    )

    print(f"[STEP 2] Training complete. Results in runs/detect/{run_name}")
    return model, results


# =========================
# STEP 3: EVALUATION
# =========================

def step3_evaluate(
    eval_dir: Path = EVAL_IMAGES_DIR,
    run_name: str = RUN_NAME,
    img_size: int = IMG_SIZE,
    device: int | str = DEVICE,
):
    """
    Use trained YOLO model to run predictions on the 3 evaluation images.

    - Loads runs/detect/<run_name>/weights/best.pt
    - Uses model.predict() on each image
    - Saves annotated images to runs/detect/<run_name>_eval/
    """
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation folder not found at {eval_dir}")

    # Get best weights from training
    best_weights = MODEL_WEIGHTS_DIR / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(
            f"best.pt not found at {best_weights}. "
            f"Make sure training finished successfully."
        )

    print(f"[STEP 3] Loading trained model weights: {best_weights}")
    model = YOLO(str(best_weights))

    # Collect images (jpg, jpeg, png)
    eval_images = sorted(
        [p for p in eval_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    if not eval_images:
        raise RuntimeError(f"No evaluation images found in {eval_dir}")

    print(f"[STEP 3] Found {len(eval_images)} evaluation images.")

    # Use a dedicated run name for eval outputs
    eval_run_name = f"{run_name}_eval"

    for img_path in eval_images:
        print(f"[STEP 3] Predicting on {img_path.name} ...")
        # Using model.predict() as required by assignment
        model.predict(
            source=str(img_path),
            imgsz=img_size,
            conf=0.25,        # adjust if needed
            device=device,
            save=True,
            project=str(MODEL_WEIGHTS_DIR),
            name=eval_run_name,
        )

    print(
        f"[STEP 3] Evaluation complete. Annotated images saved to "
        f"{(MODEL_WEIGHTS_DIR / eval_run_name).resolve()}"
    )


# =========================
# MAIN ENTRYPOINT
# =========================

def main():
    # STEP 1: Mask PCB from background
    print("========== STEP 1: OBJECT MASKING ==========")
    step1_mask_motherboard()

    # STEP 2: Train YOLOv11 on PCB dataset
    print("\n========== STEP 2: YOLOv11 TRAINING ==========")
    model, _ = step2_train_yolo()

    # STEP 3: Evaluate three test images
    print("\n========== STEP 3: EVALUATION ==========")
    step3_evaluate()


if __name__ == "__main__":
    main()
