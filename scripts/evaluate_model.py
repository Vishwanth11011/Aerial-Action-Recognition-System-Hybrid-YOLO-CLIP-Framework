import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import clip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
from itertools import cycle

# --- CONFIGURATION ---
DATA_DIR = "../datasets/clip_dataset/val" # Using validation set for evaluation
MODEL_PATH = "../models/clip_action/action_adapter.pth"
RESULTS_DIR = "../results"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MODEL DEFINITION (Must match training) ---
class ActionCLIP(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        input_dim = 512 
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes) 
        )

    def forward(self, image):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features = image_features.float()
        logits = self.adapter(image_features)
        return logits

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"-> Confusion Matrix saved to {save_path}")
    plt.close()

def plot_roc_curves(y_test, y_score, class_names, save_path):
    # Binarize the output
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta'])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    print(f"-> ROC Curves saved to {save_path}")
    plt.close()

def evaluate():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"-> Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    # Load CLIP backbone
    model_clip, preprocess = clip.load("ViT-B/32", device=DEVICE)
    
    # Initialize Model
    model = ActionCLIP(model_clip, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load Data
    print(f"-> Loading Test Data from {DATA_DIR}...")
    dataset = datasets.ImageFolder(DATA_DIR, transform=preprocess)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Ensure dataset classes match model classes
    # (If validation set is missing classes the model knows, we need to handle mapping)
    dataset_classes = dataset.classes
    print(f"-> Dataset Classes: {dataset_classes}")
    print(f"-> Model Classes:   {class_names}")

    all_preds = []
    all_labels = []
    all_probs = []

    print("-> Running Inference...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # --- METRICS ---
    
    # 1. Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n=== OVERALL ACCURACY: {acc*100:.2f}% ===\n")

    # 2. Classification Report (F1, Precision, Recall)
    # Note: If dataset classes differ from model classes, we need to map indices
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("--- Classification Report ---")
    print(report)
    
    # Save text report
    with open(f"{RESULTS_DIR}/metrics_report.txt", "w") as f:
        f.write(f"Overall Accuracy: {acc*100:.2f}%\n\n")
        f.write(report)

    # 3. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, f"{RESULTS_DIR}/confusion_matrix.png")

    # 4. ROC Curves
    # (Only works if we have enough data points for all classes)
    try:
        plot_roc_curves(all_labels, all_probs, class_names, f"{RESULTS_DIR}/roc_curves.png")
    except Exception as e:
        print(f"Could not plot ROC curves (likely missing classes in test set): {e}")

    print("\n-> Evaluation Complete. Check the 'results' folder.")

if __name__ == "__main__":
    evaluate()