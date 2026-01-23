import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# My imports
from anchors import build_all_anchors
from model import SSDModel
from gt_matching import match_anchors_to_gt
from loss import ssd_loss
from dataloader import DetectionDataset, samples, transform
from dataloader import ssd_collate_fn
from backbone import SimpleSSDBackbone

# Configs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

NUM_CLASSES = 6          # object classes (NO background)
ANCHORS_PER_CELL = 3
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4

# -----------------------
# DATA
# -----------------------
dataset = DetectionDataset(
    samples=samples,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=ssd_collate_fn
)

# -----------------------
# MODEL
# -----------------------

# Example backbone outputs 3 feature maps
# feature_channels must MATCH backbone outputs
feature_channels = [256, 256, 256]

backbone = SimpleSSDBackbone()   # 👈 your CNN backbone

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

# -----------------------
# TRAIN LOOP
# -----------------------
for epoch in range(EPOCHS):
    print("Starting epoch", epoch+1)
    model.train()

    epoch_loss = 0.0
    epoch_loc = 0.0
    epoch_conf = 0.0

    for images, gt_labels_list, gt_boxes_list in loader:
        images = images.to(DEVICE)
        # print("Images moved to device")

        # -----------------------
        # FORWARD
        # -----------------------
        cls_logits, bbox_preds = model(images)
        # print("Model forward done")
        # cls_logits: [B, A, C+1]
        # bbox_preds: [B, A, 4]

        B = images.size(0)

        total_loss = 0.0
        loc_loss_sum = 0.0
        conf_loss_sum = 0.0

        # -----------------------
        # PER-IMAGE MATCHING
        # -----------------------
        for i in range(B):
            # print(f"Matching image {i}")
            cls_targets, bbox_targets, labels_mask = match_anchors_to_gt(
                anchors=anchors,
                gt_boxes=gt_boxes_list[i].to(DEVICE),
                gt_labels=gt_labels_list[i].to(DEVICE),
                pos_threshold=0.5,
                neg_threshold=0.4,
                bg_label=0
            )
            # print(f"Matching done for image {i}")

            loss, loc_l, conf_l = ssd_loss(
                cls_logits=cls_logits[i],
                bbox_preds=bbox_preds[i],
                cls_targets=cls_targets,
                bbox_targets=bbox_targets,
                labels_mask=labels_mask,
                neg_pos_ratio=3
            )

            total_loss += loss
            loc_loss_sum += loc_l
            conf_loss_sum += conf_l

        # -----------------------
        # BACKWARD
        # -----------------------
        total_loss = total_loss / B

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_loc += (loc_loss_sum / B).item()
        epoch_conf += (conf_loss_sum / B).item()
        
    # -----------------------
    # LOGGING
    # -----------------------
    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {epoch_loss:.4f} | "
        f"Loc: {epoch_loc:.4f} | "
        f"Conf: {epoch_conf:.4f}"
    )
    
# -----------------------
# SAVE MODEL
# -----------------------
torch.save(model.state_dict(), "ssd_model.pth")
print("Training complete. Model saved.")