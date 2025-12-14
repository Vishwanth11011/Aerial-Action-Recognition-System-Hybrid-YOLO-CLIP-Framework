import os
import cv2
import csv
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_DATA_PATH = "../raw" 
SUB_DIRS = ["atomic", "Human_Human", "Human_Object"]

OUTPUT_YOLO_PATH = "../datasets/yolo_dataset_tiled"
OUTPUT_CLIP_PATH = "../datasets/clip_dataset"
TRAIN_RATIO = 0.8

# TILING CONFIGURATION
SLICE_H = 640
SLICE_W = 640
OVERLAP_H = 0.2
OVERLAP_W = 0.2

# --- UPDATED ACTION MAPPING ---
# We allow both the String Name and the ID (just in case)
ACTION_MAP = {
    0: "Carrying",
    1: "Drinking",
    2: "Handshaking",
    3: "Hugging",
    4: "Kicking",
    5: "Lying",
    6: "Punching",
    7: "Reading",
    8: "Running",
    9: "Sitting",
    10: "Standing",
    11: "Walking",
    12: "Waving"
}

# Create a set of valid strings for fast lookup
VALID_ACTIONS = set(ACTION_MAP.values())

def get_action_name(raw_label):
    """
    Normalizes the action label. 
    Handles: "Sitting" (String), "sitting" (lowercase), or "9" (String ID)
    """
    # 1. Check if it's a digit (e.g., "9")
    if raw_label.isdigit():
        idx = int(raw_label)
        if idx in ACTION_MAP:
            return ACTION_MAP[idx]
            
    # 2. Check if it's a string key (e.g., "Sitting" or "sitting")
    clean_label = raw_label.strip().capitalize() # Force "sitting" -> "Sitting"
    if clean_label in VALID_ACTIONS:
        return clean_label
        
    return None # Invalid action

def compute_slices(img_h, img_w, slice_h, slice_w, overlap_h, overlap_w):
    """Generates coordinates (x1, y1, x2, y2) for sliding windows"""
    step_h = int(slice_h * (1 - overlap_h))
    step_w = int(slice_w * (1 - overlap_w))
    
    slices = []
    for y in range(0, img_h, step_h):
        for x in range(0, img_w, step_w):
            y2 = min(y + slice_h, img_h)
            x2 = min(x + slice_w, img_w)
            # Adjust start point if we hit the edge to ensure fixed size
            y1 = y2 - slice_h if y2 - slice_h >= 0 else 0
            x1 = x2 - slice_w if x2 - slice_w >= 0 else 0
            slices.append([x1, y1, x2, y2])
    return slices

def prepare_dirs():
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_YOLO_PATH}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_YOLO_PATH}/labels/{split}", exist_ok=True)
        # Create folders for all 13 classes
        for action in VALID_ACTIONS:
            os.makedirs(f"{OUTPUT_CLIP_PATH}/{split}/{action}", exist_ok=True)

def process_dataset():
    prepare_dirs()
    
    for category in SUB_DIRS:
        category_path = os.path.join(RAW_DATA_PATH, category)
        frames_dir = os.path.join(category_path, "extracted_frames")
        ann_dir = os.path.join(category_path, "annotations_final")
        
        if not os.path.exists(frames_dir) or not os.path.exists(ann_dir):
            print(f"Skipping {category}: Folders not found.")
            continue
            
        print(f"Processing Category: {category}...")
        
        image_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files, desc=f"Slicing {category}"):
            # 1. Match Image to Annotation File
            file_stem = Path(img_file).stem 
            ann_file = f"{file_stem}.txt"
            ann_path = os.path.join(ann_dir, ann_file)
            img_path = os.path.join(frames_dir, img_file)
            
            if not os.path.exists(ann_path):
                continue

            # 2. Read Annotations
            boxes = [] 
            with open(ann_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    # CSV Format: Class, Action, ID, X, Y, W, H
                    try:
                        raw_action = row[1]
                        
                        # VALIDATE ACTION using the new logic
                        action_name = get_action_name(raw_action)
                        if action_name is None:
                            continue # Skip rows that aren't in our 13 classes
                            
                        x, y, w, h = float(row[3]), float(row[4]), float(row[5]), float(row[6])
                        boxes.append({'action': action_name, 'box': [x, y, w, h]})
                    except (ValueError, IndexError):
                        continue

            if not boxes: continue

            # 3. Load Image
            img = cv2.imread(img_path)
            if img is None: continue
            img_h, img_w = img.shape[:2]
            
            split = 'train' if random.random() < TRAIN_RATIO else 'val'

            # --- A. GENERATE CLIP CROPS (Using standardized action names) ---
            for i, item in enumerate(boxes):
                act = item['action']
                bx, by, bw, bh = item['box']
                
                # Crop logic
                x1, y1 = max(0, int(bx)), max(0, int(by))
                x2, y2 = min(img_w, int(bx+bw)), min(img_h, int(by+bh))
                
                if x2 > x1 and y2 > y1:
                    crop = img[y1:y2, x1:x2]
                    crop_name = f"{category}_{file_stem}_{i}_{act}.jpg"
                    cv2.imwrite(f"{OUTPUT_CLIP_PATH}/{split}/{act}/{crop_name}", crop)

            # --- B. GENERATE YOLO TILES ---
            slices = compute_slices(img_h, img_w, SLICE_H, SLICE_W, OVERLAP_H, OVERLAP_W)
            
            for i, (sx1, sy1, sx2, sy2) in enumerate(slices):
                valid_yolo_labels = []
                
                for item in boxes:
                    bx, by, bw, bh = item['box']
                    bx2, by2 = bx + bw, by + bh
                    
                    ix1 = max(sx1, bx)
                    iy1 = max(sy1, by)
                    ix2 = min(sx2, bx2)
                    iy2 = min(sy2, by2)
                    
                    if ix2 > ix1 and iy2 > iy1:
                        new_x_min = ix1 - sx1
                        new_y_min = iy1 - sy1
                        new_w = ix2 - ix1
                        new_h = iy2 - iy1
                        
                        nx_center = (new_x_min + new_w / 2) / SLICE_W
                        ny_center = (new_y_min + new_h / 2) / SLICE_H
                        nw = new_w / SLICE_W
                        nh = new_h / SLICE_H
                        
                        nx_center = min(max(nx_center, 0), 1)
                        ny_center = min(max(ny_center, 0), 1)
                        nw = min(max(nw, 0), 1)
                        nh = min(max(nh, 0), 1)

                        # Write label: "0" because YOLO just needs to know "Person is here"
                        valid_yolo_labels.append(f"0 {nx_center:.6f} {ny_center:.6f} {nw:.6f} {nh:.6f}")
                
                if valid_yolo_labels:
                    slice_img = img[sy1:sy2, sx1:sx2]
                    unique_name = f"{category}_{file_stem}_slice_{i}"
                    
                    save_img_path = f"{OUTPUT_YOLO_PATH}/images/{split}/{unique_name}.jpg"
                    cv2.imwrite(save_img_path, slice_img)
                    
                    save_lbl_path = f"{OUTPUT_YOLO_PATH}/labels/{split}/{unique_name}.txt"
                    with open(save_lbl_path, 'w') as f:
                        f.write('\n'.join(valid_yolo_labels))

    print("Processing Complete.")
    print(f"YOLO Dataset: {OUTPUT_YOLO_PATH}")
    print(f"CLIP Dataset (13 Classes): {OUTPUT_CLIP_PATH}")

if __name__ == "__main__":
    process_dataset()