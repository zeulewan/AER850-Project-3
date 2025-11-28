"""
AER850 Project 3 – PCB Masking + YOLOv11 Training + Evaluation

- Step 1: Object masking of motherboard_image.JPEG using OpenCV
- Step 2: YOLOv11 nano training on PCB dataset using Ultralytics
- Step 3: Evaluation of images in the evaluation directory using model.predict()
"""

import os  # Standard library, currently unused but retained for potential future use

import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
from ultralytics import YOLO  # Ultralytics YOLOv11 interface

from pathlib import Path  # Object-oriented filesystem paths


# =========================
# CONFIG
# =========================

# Root directory of the project (directory containing this script)
PROJECT_ROOT = Path(__file__).resolve().parent

# Path to the motherboard image used for masking
MOTHERBOARD_IMAGE_PATH = PROJECT_ROOT / "motherboard_image.JPEG"

# Directory where masking results will be stored
MASK_OUTPUT_DIR = PROJECT_ROOT / "masking_outputs"

# Path to YOLO dataset configuration file (Ultralytics YAML format)
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"

# Root directory for YOLO runs (Ultralytics default convention)
MODEL_WEIGHTS_DIR = PROJECT_ROOT / "runs" / "detect"

# Directory containing evaluation images for final prediction
EVAL_IMAGES_DIR = PROJECT_ROOT / "data" / "evaluation"

# YOLO training configuration
RUN_NAME = "pcb_yolo11n"   # Name used for the training run subdirectory
EPOCHS = 150               # Number of training epochs
BATCH_SIZE = 12            # Training batch size
IMG_SIZE = 1024            # Input image size for YOLO (square)
DEVICE = 0                 # CUDA device index (0 selects the first GPU)


# =========================
# UTILS
# =========================

def ensure_dir(path: Path):
    """
    Ensure that a directory exists; create it along with parents if necessary.
    """
    path.mkdir(parents=True, exist_ok=True)


# =========================
# STEP 1: OBJECT MASKING
# =========================

def step1_mask_motherboard(
    input_path: Path = MOTHERBOARD_IMAGE_PATH,
    output_dir: Path = MASK_OUTPUT_DIR,
):
    """
    Perform PCB masking on the input motherboard image.

    Processing pipeline:
    1. Load original RGB image (in BGR format as used by OpenCV)
    2. Convert to grayscale
    3. Apply Gaussian blur to reduce noise
    4. Apply thresholding to obtain a binary image
    5. Compute Canny edges for visualization
    6. Locate the largest external contour as PCB outline
    7. Create a binary mask from this contour
    8. Extract PCB region via bitwise masking
    9. Save intermediate and final results for reporting
    """
    # Ensure that the directory for masking outputs exists
    ensure_dir(output_dir)

    # Validate that the input motherboard image exists
    if not input_path.exists():
        raise FileNotFoundError(f"Motherboard image not found at {input_path}")

    # Load the original image as BGR (default OpenCV format)
    orig_bgr = cv2.imread(str(input_path))
    if orig_bgr is None:
        raise ValueError(f"Failed to read image at {input_path}")

    # Rotate 90 degrees clockwise (to the right)
    orig_bgr = cv2.rotate(orig_bgr, cv2.ROTATE_90_CLOCKWISE)

    # Convert BGR image to grayscale for subsequent processing
    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce high-frequency noise and improve thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Manual thresholding on the blurred image (tune 'thresh_value' as needed)
    thresh_value = 120  # can be adjusted: 110, 130, 150, etc.
    _, thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)

    # Check polarity: ensure PCB region becomes white (foreground)
    white_ratio = np.mean(thresh == 255)
    if white_ratio < 0.5:
        # If white area is too small, invert so the board becomes foreground
        thresh = cv2.bitwise_not(thresh)

    # Create a structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # Perform morphological closing to fill gaps and smooth the PCB region
    thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Compute Canny edges for visualization of detected edges
    edges = cv2.Canny(blurred, 100, 200)

    # Find external contours on the processed binary image
    contours, _ = cv2.findContours(
        thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Abort if no contours are detected (indicates a preprocessing issue)
    if not contours:
        raise RuntimeError("No contours found – thresholding parameters may require adjustment.")

    # Select the contour with maximum area as the PCB outline
    largest_contour = max(contours, key=cv2.contourArea)

    # Initialize an empty mask (single-channel, same size as input)
    mask = np.zeros_like(gray)
    # Fill the largest contour area on the mask with white (255)
    cv2.drawContours(mask, [largest_contour], contourIdx=-1, color=255, thickness=cv2.FILLED)

    # Optionally refine the mask via morphological opening to remove small artifacts
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Apply the binary mask to the original image to isolate the PCB
    extracted_bgr = cv2.bitwise_and(orig_bgr, orig_bgr, mask=mask)

    # Persist intermediate and final images for documentation and analysis
    cv2.imwrite(str(output_dir / "01_gray.png"), gray)
    cv2.imwrite(str(output_dir / "02_thresh.png"), thresh)
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
    Train a YOLOv11 nano detection model on the PCB dataset.

    Ultralytics automatically creates runs in:
        runs/detect/<run_name>/
    containing:
        - Weights: best.pt, last.pt
        - Training metrics and plots
    """
    # Verify that the dataset YAML exists at the configured location
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found at {data_yaml}")

    # Instantiate YOLOv11 nano model (weights automatically downloaded if missing)
    print("[STEP 2] Loading YOLOv11 nano model (yolo11n.pt)...")
    model = YOLO("yolo11n.pt")

    # Launch training with specified hyperparameters
    print("[STEP 2] Starting training...")
    results = model.train(
        data=str(data_yaml),   # Path to dataset configuration YAML
        epochs=epochs,         # Number of training epochs
        imgsz=img_size,        # Input spatial resolution
        batch=batch_size,      # Training batch size
        device=device,         # Compute device (GPU index or 'cpu')
        name=run_name,         # Run name used for output directory
        # Optional arguments left commented for clarity and potential extension:
        # workers=8,           # Number of data loading workers
        # patience=50,         # Early stopping patience
    )

    print(f"[STEP 2] Training complete. Results in runs/detect/{run_name}")
    return model, results  # Return model and training results object


# =========================
# STEP 3: EVALUATION
# =========================

def step3_evaluate(
    eval_dir: Path = EVAL_IMAGES_DIR,
    run_name: str = RUN_NAME,
    img_size: int = IMG_SIZE,
    device: int | str = DEVICE,
    font_size: float = 0.4,   # smaller label text
    line_width: int = 1,      # thinner boxes
):
    """
    Evaluate the trained YOLO model on a set of evaluation images.

    Evaluation procedure:
    - Load best-performing weights from training run
    - Run prediction on each image in the evaluation directory
    - Manually draw and save annotated predictions with custom font size and line width
    """
    # Ensure that the evaluation directory is available
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation folder not found at {eval_dir}")

    # Construct the path to the best model weights within the YOLO run directory
    best_weights = MODEL_WEIGHTS_DIR / run_name / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(
            f"best.pt not found at {best_weights}. "
            f"Ensure that training completed successfully."
        )

    # Load the trained YOLO model from the best-performing weights
    print(f"[STEP 3] Loading trained model weights: {best_weights}")
    model = YOLO(str(best_weights))

    # Collect all evaluation images with supported file extensions
    eval_images = sorted(
        [p for p in eval_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )

    # Raise an error if no evaluation images are present
    if not eval_images:
        raise RuntimeError(f"No evaluation images found in {eval_dir}")

    print(f"[STEP 3] Found {len(eval_images)} evaluation images.")

    # Name for the evaluation run; separates prediction outputs from training artifacts
    eval_run_name = f"{run_name}_eval"
    eval_output_dir = MODEL_WEIGHTS_DIR / eval_run_name
    ensure_dir(eval_output_dir)

    # Run inference on each evaluation image
    for img_path in eval_images:
        print(f"[STEP 3] Predicting on {img_path.name} ...")

        # Run model.predict WITHOUT unsupported args like font_size / line_width
        # and do not auto-save; we will save manually after plotting.
        results = model.predict(
            source=str(img_path),
            imgsz=img_size,
            conf=0.25,
            device=device,
            save=False,
        )

        # results is usually a list of Result objects
        for i, r in enumerate(results):
            # Draw boxes/labels with custom style.
            # r.plot returns an annotated BGR image as a NumPy array.
            annotated = r.plot(
                line_width=line_width,
                font_size=font_size,
            )

            # If there are multiple frames, avoid overwriting by using an index
            if len(results) > 1:
                out_name = f"{img_path.stem}_{i}{img_path.suffix}"
            else:
                out_name = img_path.name

            out_path = eval_output_dir / out_name
            cv2.imwrite(str(out_path), annotated)

    # Print final location of evaluation outputs
    print(
        f"[STEP 3] Evaluation complete. Annotated images saved to "
        f"{eval_output_dir.resolve()}"
    )


# =========================
# MAIN ENTRYPOINT
# =========================

def main():
    """
    Main execution pipeline:
    1. Perform PCB masking on the motherboard image.
    2. Train YOLOv11 nano on the configured PCB dataset.
    3. Evaluate the trained model on the evaluation image set.
    """
    # # Execute Step 1: PCB object masking
    # print("========== STEP 1: OBJECT MASKING ==========")
    # step1_mask_motherboard()

    # # Execute Step 2: YOLOv11 model training
    # print("\n========== STEP 2: YOLOv11 TRAINING ==========")
    # model, _ = step2_train_yolo()

    # Execute Step 3: Model evaluation on selected images
    print("\n========== STEP 3: EVALUATION ==========")
    step3_evaluate(font_size=1, line_width=7)


# Execute main pipeline when script is run as the primary module
if __name__ == "__main__":
    main()
