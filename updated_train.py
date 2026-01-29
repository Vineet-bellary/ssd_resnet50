import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

# My imports
from anchors import build_all_anchors
from model import SSDModel
from loss import ssd_loss
from dataloader import (
    DetectionDataset,
    train_samples,
    valid_samples,
    transform,
    ssd_collate_fn
)
from new_backbone import ResNet50Backbone

# Global Configs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
ANCHORS_PER_CELL = 6
BATCH_SIZE = 16
EPOCHS = 10  # Increased epochs to show value of checkpointing
LR = 1e-4
CHECKPOINT_PATH = "checkpoint.pth"

def main():
    print(f"Starting training on {DEVICE} with num_workers=2...")

    # 1. Create Anchors
    anchors_cpu = build_all_anchors()
    anchors_gpu = anchors_cpu.to(DEVICE) 

    # 2. Setup DataLoaders
    train_dataset = DetectionDataset(train_samples, anchors_cpu, transform)
    valid_dataset = DetectionDataset(valid_samples, anchors_cpu, transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=ssd_collate_fn, 
        num_workers=2,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=ssd_collate_fn, 
        num_workers=2,
        pin_memory=True
    )

    # 3. Model Setup
    backbone = ResNet50Backbone()
    model = SSDModel(backbone, [512, 1024, 2048], ANCHORS_PER_CELL, NUM_CLASSES)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- CHECKPOINT LOADING ---
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # 4. Training Loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        for images, cls_targets, bbox_targets, labels_mask in train_loader:
            images = images.to(DEVICE)
            cls_targets = cls_targets.to(DEVICE)
            bbox_targets = bbox_targets.to(DEVICE)
            labels_mask = labels_mask.to(DEVICE)

            cls_logits, bbox_preds = model(images)

            loss, _, _ = ssd_loss(
                cls_logits, bbox_preds, 
                cls_targets, bbox_targets, labels_mask
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 5. Validation Loop
        model.eval()
        val_accumulator = 0.0
        with torch.no_grad():
            for images, cls_targets, bbox_targets, labels_mask in valid_loader:
                images, cls_targets = images.to(DEVICE), cls_targets.to(DEVICE)
                bbox_targets, labels_mask = bbox_targets.to(DEVICE), labels_mask.to(DEVICE)

                cls_logits, bbox_preds = model(images)
                loss, _, _ = ssd_loss(cls_logits, bbox_preds, cls_targets, bbox_targets, labels_mask)
                val_accumulator += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_accumulator / len(valid_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- SAVE CHECKPOINT ---
        # Save every epoch so you never lose more than one epoch of work
        checkpoint_data = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    # Save final model
    torch.save(model.state_dict(), "ssd_model_final.pth")
    print("Training complete.")

if __name__ == "__main__":
    main()