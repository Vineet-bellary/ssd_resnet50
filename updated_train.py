import torch
from torch.utils.data import DataLoader
import os

from anchors import build_all_anchors
from model import SSDModel
from loss import ssd_loss
from dataloader import (
    DetectionDataset,
    load_samples,
    transform,
    ssd_collate_fn
)
from new_backbone import ResNet50Backbone


# -------------------------------------------------
# Global Configs
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 4
ANCHORS_PER_CELL = 6
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
CHECKPOINT_PATH = "checkpoint.pth"


# -------------------------------------------------
# Dataset paths
# -------------------------------------------------
TRAIN_IMAGE_DIR = r"ssd-object-detection-5\train"
TRAIN_JSON = "preprocessed_weapon_train.json"

VALID_IMAGE_DIR = r"ssd-object-detection-5\valid"
VALID_JSON = "preprocessed_weapon_valid.json"


def main():
    print(f"Starting training on {DEVICE}...")

    # -------------------------------------------------
    # 1. Anchors
    # -------------------------------------------------
    anchors_cpu = build_all_anchors()
    anchors_gpu = anchors_cpu.to(DEVICE)

    print(f"[INFO] Total anchors: {anchors_cpu.shape[0]}")
    
    # -------------------------------------------------
    # Load samples
    # -------------------------------------------------
    train_samples = load_samples(TRAIN_JSON, TRAIN_IMAGE_DIR)
    valid_samples = load_samples(VALID_JSON, VALID_IMAGE_DIR)

    # -------------------------------------------------
    # 2. DataLoaders
    # -------------------------------------------------
    train_dataset = DetectionDataset(train_samples, anchors_cpu, transform)
    valid_dataset = DetectionDataset(valid_samples, anchors_cpu, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=ssd_collate_fn,
        num_workers=0,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=ssd_collate_fn,
        num_workers=0,
        pin_memory=True
    )

    # -------------------------------------------------
    # 3. Model
    # -------------------------------------------------
    backbone = ResNet50Backbone()
    model = SSDModel(backbone, [512, 1024, 2048], ANCHORS_PER_CELL, NUM_CLASSES)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -------------------------------------------------
    # 4. Checkpoint load
    # -------------------------------------------------
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1

    debug_done = False

    # -------------------------------------------------
    # 5. Training loop
    # -------------------------------------------------
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        for images, cls_targets, bbox_targets, labels_mask in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            cls_targets = cls_targets.to(DEVICE, non_blocking=True)
            bbox_targets = bbox_targets.to(DEVICE, non_blocking=True)
            labels_mask = labels_mask.to(DEVICE, non_blocking=True)

            cls_logits, bbox_preds = model(images)

            # 🔎 ONE-TIME DEBUG CHECK
            if not debug_done:
                print("\n[DEBUG CHECK]")
                print("Images:", images.shape)
                print("Cls logits:", cls_logits.shape)
                print("BBox preds:", bbox_preds.shape)
                print("Unique labels:", torch.unique(cls_targets))
                print("Labels mask values:", torch.unique(labels_mask))
                debug_done = True

            loss, _, _ = ssd_loss(
                cls_logits,
                bbox_preds,
                cls_targets,
                bbox_targets,
                labels_mask
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, cls_targets, bbox_targets, labels_mask in valid_loader:
                images = images.to(DEVICE)
                cls_targets = cls_targets.to(DEVICE)
                bbox_targets = bbox_targets.to(DEVICE)
                labels_mask = labels_mask.to(DEVICE)

                cls_logits, bbox_preds = model(images)
                loss, _, _ = ssd_loss(
                    cls_logits,
                    bbox_preds,
                    cls_targets,
                    bbox_targets,
                    labels_mask
                )
                val_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            CHECKPOINT_PATH
        )

    torch.save(model.state_dict(), "ssd_model_final.pth")
    print("Training complete.")


if __name__ == "__main__":
    main()
