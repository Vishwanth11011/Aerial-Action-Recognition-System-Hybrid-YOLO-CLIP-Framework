import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import clip
from tqdm import tqdm
import os
import shutil

# --- CONFIGURATION ---
DATA_DIR = "../datasets/clip_dataset"
MODEL_SAVE_PATH = "../models/clip_action/action_adapter.pth"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ActionCLIP(nn.Module):
    """
    Custom Architecture:
    1. Visual Encoder (CLIP ViT) - Frozen
    2. Visual Adapter (Trainable)
    """
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        
        # Freeze CLIP weights
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Feature dim for ViT-B/32 is 512
        input_dim = 512 
        
        # Trainable Adapter
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

def prune_empty_folders(path):
    """
    Scans the directory and removes any class folders that are empty.
    This prevents ImageFolder from crashing.
    """
    if not os.path.exists(path): return

    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    removed_count = 0
    
    for folder in folders:
        folder_path = os.path.join(path, folder)
        # Check if folder is empty (no files)
        if not os.listdir(folder_path):
            print(f"   [Warning] Removing empty class folder: {folder}")
            os.rmdir(folder_path) # Safe delete (only works if empty)
            removed_count += 1
            
    if removed_count > 0:
        print(f"   -> Pruned {removed_count} empty classes from {path}")

def train_action_model():
    print(f"-> Using Device: {DEVICE}")
    
    # 1. Clean Data (Fixes the FileNotFoundError)
    print("-> Checking dataset for empty classes...")
    prune_empty_folders(os.path.join(DATA_DIR, 'train'))
    prune_empty_folders(os.path.join(DATA_DIR, 'val'))

    # 2. Load CLIP
    print("-> Loading CLIP...")
    model_clip, preprocess = clip.load("ViT-B/32", device=DEVICE)
    
    # 3. Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        preprocess 
    ])
    
    val_transform = preprocess

    print("-> Loading Datasets...")
    # ImageFolder will now only load classes that actually have images
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transform)
    
    # Dynamic Class Detection
    found_classes = train_dataset.classes
    num_classes = len(found_classes)
    print(f"-> Valid Classes Found ({num_classes}): {found_classes}")
    
    if num_classes == 0:
        print("Error: No classes found! Check your dataset generation.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 4. Initialize Model with correct number of classes
    model = ActionCLIP(model_clip, num_classes=num_classes).to(DEVICE)
    
    optimizer = optim.Adam(model.adapter.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Handle case where validation set might be empty for very small datasets
        if val_total > 0:
            val_acc = 100 * val_correct / val_total
        else:
            val_acc = 0.0
            
        print(f"-> Epoch {epoch+1} Results: Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc >= best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            # Save the class names too so inference knows which index is which action
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': found_classes, 
                'clip_version': "ViT-B/32"
            }, MODEL_SAVE_PATH)
            print(f"   [Saved Best Model to {MODEL_SAVE_PATH}]")

    print("-> Action Recognition Training Complete.")

if __name__ == "__main__":
    train_action_model()