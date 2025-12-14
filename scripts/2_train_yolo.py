from ultralytics import YOLO
import os

# --- CONFIGURATION ---
DATASET_PATH = os.path.abspath("../datasets/yolo_dataset_tiled")
MODEL_SAVE_DIR = "../models/yolo"
PRETRAINED_WEIGHTS = "yolo11s.pt"  # s=small, m=medium (use 's' for speed, 'm' for better accuracy)

def create_yaml():
    """Generates the data.yaml file required by YOLO"""
    yaml_content = f"""
path: {DATASET_PATH}
train: images/train
val: images/val

# We only have 1 class for detection: Person
# The Action Recognition model handles the specific action later.
nc: 1
names: ['person']
"""
    with open("custom_data.yaml", "w") as f:
        f.write(yaml_content)
    print("-> custom_data.yaml created.")

def train_yolo():
    create_yaml()
    
    # Initialize Model
    model = YOLO(PRETRAINED_WEIGHTS)
    
    print(f"-> Starting training on {DATASET_PATH}...")
    
    # Train
    results = model.train(
        data='custom_data.yaml',
        epochs=50,                  # 50 is usually sufficient for fine-tuning
        imgsz=640,                  # MUST match the slice size from Step 1
        batch=16,                   # Reduce to 8 if you run out of GPU memory
        project=MODEL_SAVE_DIR,
        name='drone_person_detector',
        patience=10,                # Stop if no improvement for 10 epochs
        device=0,                   # Use 0 for GPU, 'cpu' for CPU
        exist_ok=True,
        augment=True                # robust augmentation helps with drone variances
    )
    
    print("-> YOLO Training Finished.")
    print(f"-> Best model saved at: {MODEL_SAVE_DIR}/drone_person_detector/weights/best.pt")

if __name__ == "__main__":
    train_yolo()