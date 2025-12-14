import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
import supervision as sv
from ultralytics import YOLO # Using the stable YOLO engine
import clip
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURATION ---
TEST_DIR = "../test"
OUTPUT_DIR = "../results/test_inference_OPTIMIZED_NATIVE"

# 1. MAXIMIZED RESOLUTION (Improves detection of small objects)
INFERENCE_SIZE = 1536 
CONFIDENCE_THRESHOLD = 0.15 # Trade-off between accuracy and noise

# 2. CONTEXT AWARENESS (Improves CLIP Classification)
CROP_PADDING = 0.15 # 15% padding around the box

# Model Paths
YOLO_PATH = "../models/yolo/drone_person_detector/weights/best.pt"
ACTION_MODEL_PATH = "../models/clip_action/action_adapter.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MODEL DEFINITIONS (Required for loading weights) ---
class ActionCLIP(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        for param in self.clip_model.parameters(): param.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, num_classes)
        )
    def forward(self, image):
        with torch.no_grad():
            features = self.clip_model.encode_image(image).float()
        return self.adapter(features)

def inference_native_optimized():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load CLIP and Action Model
    try:
        checkpoint = torch.load(ACTION_MODEL_PATH, map_location=DEVICE)
        class_names = checkpoint['class_names']
        model_clip, preprocess = clip.load("ViT-B/32", device=DEVICE)
        action_model = ActionCLIP(model_clip, len(class_names)).to(DEVICE)
        action_model.load_state_dict(checkpoint['model_state_dict'])
        action_model.eval()
    except Exception as e:
        print(f"FATAL ERROR: Failed to load models. Check paths: {e}"); return

    # 2. Load YOLO Detection Model
    try:
        yolo_model = YOLO(YOLO_PATH)
    except Exception as e:
        print(f"FATAL ERROR: Failed to load YOLO model. Check path: {e}"); return
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    
    image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"-> Starting Optimized Native Inference (Size: {INFERENCE_SIZE})...")

    for img_file in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(TEST_DIR, img_file)
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # 3. Standard YOLO Detection (Fast Inference, High Resolution)
        results = yolo_model.predict(
            source=frame,
            imgsz=INFERENCE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            device=DEVICE,
            verbose=False
        )
        
        boxes = results[0].boxes
        if not boxes or len(boxes) == 0:
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_file), frame)
            continue
            
        # Convert to Supervision Detections
        detections = sv.Detections(
            xyxy=boxes.xyxy.cpu().numpy(),
            confidence=boxes.conf.cpu().numpy(),
            class_id=boxes.cls.cpu().numpy().astype(int)
        )
        
        # 4. Prepare Batch for CLIP Inference with Padding
        crops_to_process = []
        
        for box in detections.xyxy:
            x1, y1, x2, y2 = map(int, box)
            
            # --- CONTEXT PADDING LOGIC ---
            w_box, h_box = x2 - x1, y2 - y1
            pad_w, pad_h = int(w_box * CROP_PADDING), int(h_box * CROP_PADDING)
            
            # Apply padding and clamp to image bounds
            x1_p = max(0, x1 - pad_w)
            y1_p = max(0, y1 - pad_h)
            x2_p = min(frame.shape[1], x2 + pad_w)
            y2_p = min(frame.shape[0], y2 + pad_h)
            # -----------------------------
            
            person_crop = frame[y1_p:y2_p, x1_p:x2_p]
            pil_img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
            crops_to_process.append(preprocess(pil_img).unsqueeze(0))

        # 5. Run Batch Inference
        if crops_to_process:
            batch_tensors = torch.cat(crops_to_process).to(DEVICE)
            with torch.no_grad():
                logits = action_model(batch_tensors)
                _, pred_idxs = torch.max(logits, 1)
            
            # 6. Annotate
            predicted_labels = [class_names[idx.item()] for idx in pred_idxs]

            annotated_frame = box_annotator.annotate(scene=frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=predicted_labels)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_file), annotated_frame)

    print(f"\n-> Inference complete. Results in {OUTPUT_DIR}. (Note: Smallest subjects may be missed due to SAHI being disabled.)")

if __name__ == "__main__":
    inference_native_optimized()