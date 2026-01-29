import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

# My imports
from anchors import build_all_anchors
from model import SSDModel
from gt_matching import match_anchors_to_gt
from loss import ssd_loss
from dataloader import (
    DetectionDataset,
    train_samples,
    valid_samples,
    transform,
    ssd_collate_fn
)
# from backbone import SimpleSSDBackbone
from new_backbone import ResNet50Backbone

# Configs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

NUM_CLASSES = 7
ANCHORS_PER_CELL = 6
BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-4
CHECKPOINT_PATH = "checkpoint.pth"

# -----------------------
# DATA
# -----------------------
anchors_cpu = build_all_anchors()

train_dataset = DetectionDataset(
    samples=train_samples,
    anchors=anchors_cpu,
    transform=transform
)

valid_dataset = DetectionDataset(
    samples=valid_samples,
    anchors=anchors_cpu,
    transform=transform
)

# ADD THESE: You must define the loaders before the loops
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=ssd_collate_fn,
    num_workers=4  # Speed boost: matching now happens in parallel
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=ssd_collate_fn,
    num_workers=4
)

# -----------------------
# MODEL
# -----------------------

# Example backbone outputs 3 feature maps
# feature_channels must MATCH backbone outputs
feature_channels = [512, 1024, 2048]

backbone = ResNet50Backbone()

model = SSDModel(
    backbone=backbone,
    feature_channels=feature_channels,
    num_anchors=ANCHORS_PER_CELL,
    num_classes=NUM_CLASSES
)

model.to(DEVICE)

# -----------------------
# ANCHORS (ONCE)
# -----------------------
anchors = build_all_anchors()          # [A, 4]
anchors = anchors.to(DEVICE)
A = anchors.shape[0]

print("Total anchors:", A)

# -----------------------
# OPTIMIZER
# -----------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

# Checkpoint: Load model weights if needed
def save_checkpoint(epoch, model, optimizer, path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved at epoch {epoch}")
    
def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path, map_location=DEVICE)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"✅ Checkpoint loaded, resuming from epoch {start_epoch}")
    
    return start_epoch


# -----------------------
# TRAIN LOOP
# -----------------------
start_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    start_epoch = load_checkpoint(model, optimizer, path=CHECKPOINT_PATH)
    print("Checkpoint loaded. Resuming training...")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    epoch_loss = 0.0
    
    for images, cls_targets, bbox_targets, labels_mask in train_loader:
        images = images.to(DEVICE)
        cls_targets = cls_targets.to(DEVICE)
        bbox_targets = bbox_targets.to(DEVICE)
        labels_mask = labels_mask.to(DEVICE)

        cls_logits, bbox_preds = model(images)

        total_loss, _, _ = ssd_loss(
            cls_logits=cls_logits,
            bbox_preds=bbox_preds,
            cls_targets=cls_targets,
            bbox_targets=bbox_targets,
            labels_mask=labels_mask,
            neg_pos_ratio=3
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
       
    # Validation loop 
    model.eval()
    val_accumulator = 0.0 

    with torch.no_grad():
        for images, cls_targets, bbox_targets, labels_mask in valid_loader:
            images, cls_targets = images.to(DEVICE), cls_targets.to(DEVICE)
            bbox_targets, labels_mask = bbox_targets.to(DEVICE), labels_mask.to(DEVICE)

            cls_logits, bbox_preds = model(images)

            loss, _, _ = ssd_loss(
                cls_logits=cls_logits,
                bbox_preds=bbox_preds,
                cls_targets=cls_targets,
                bbox_targets=bbox_targets,
                labels_mask=labels_mask
            )
            # FIX from your screenshot: loss.item() is a float. 
            # val_accumulator is now a float. No more .item() needed later.
            val_accumulator += loss.item()

    # Final Average Calculations
    avg_train_loss = epoch_loss / len(train_loader)
    avg_val_loss = val_accumulator / len(valid_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}"
    )
    
    # Save checkpoint every 2 epochs
    if (epoch + 1) % 2 == 0:
        save_checkpoint(epoch, model, optimizer, path=CHECKPOINT_PATH)


# -----------------------
# SAVE MODEL
# -----------------------
torch.save(model.state_dict(), "ssd_model.pth")
print("Training complete. Model saved.")