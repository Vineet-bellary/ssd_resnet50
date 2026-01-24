import torch
import cv2
import numpy as np
from torchvision.ops import nms

from model import SSDModel
from anchors import build_all_anchors
# from gt_matching import decode_boxes
from backbone import SimpleSSDBackbone

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# assert DEVICE.type == "cuda", "Inference requires a CUDA-capable device."
NUM_CLASSES = 7          # object classes (NO background)
CONF_THRESH = 0.5
IOU_THRES = 0.5

# Class indices to Labels mnapping
CLASS_LABELS = {
    1: "vehicles-and-traffic-signals",
    2: "Bike",
    3: "Bus",
    4: "Car",
    5: "Person",
    6: "Traffic Signal",
    7: "Truck",
}

# LOAD MODEL
# Backbone
backbone = SimpleSSDBackbone()

# SAME values as training
FEATURE_CHANNELS = [256, 256, 256]   # channels of feature maps
NUM_ANCHORS = 3                    # anchors per cell

model = SSDModel(
    backbone=backbone,
    feature_channels=FEATURE_CHANNELS,
    num_anchors=NUM_ANCHORS,
    num_classes=NUM_CLASSES
)
state_dict = torch.load(
    "ssd_model.pth",
    map_location=DEVICE,
    weights_only=True
)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# LOAD IMAGE
img = cv2.imread(r"Object-detection-1\test\FudanPed00017_png.rf.8e81eb20ed1017355998255f9c0f0652.jpg")
h, w, _ = img.shape

img_resized = cv2.resize(img, (224, 224))       # assuming model input size is 300x300
img_tensor = torch.tensor(img_resized).permute(2,0,1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

# Anchors
anchors = build_all_anchors().to(DEVICE)    # [A, 4]

# decode boxes
def decode_boxes(anchors, bbox_preds):
    ax, ay, aw, ah = anchors[:,0], anchors[:,1], anchors[:,2], anchors[:,3]
    tx, ty, tw, th = bbox_preds[:,0], bbox_preds[:,1], bbox_preds[:,2], bbox_preds[:,3]

    cx = tx * aw + ax
    cy = ty * ah + ay
    w  = torch.exp(tw) * aw
    h  = torch.exp(th) * ah

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


# ------------------
# FORWARD
# ------------------
with torch.no_grad():
    conf_preds, loc_preds = model(img_tensor)

# ------------------
# POST PROCESS
# ------------------
conf_scores = torch.softmax(conf_preds[0], dim=1)
scores, labels = conf_scores.max(dim=1)

mask = (scores > CONF_THRESH) & (labels > 0)
scores = scores[mask]
labels = labels[mask]
loc_preds = loc_preds[0][mask]
anchors = anchors[mask]

boxes = decode_boxes(anchors, loc_preds)
boxes = torch.clamp(boxes, 0, 1)

# scale back to original image
boxes[:, [0,2]] *= w
boxes[:, [1,3]] *= h

boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)

# NMS
final_boxes = []
final_scores = []
final_labels = []

for cls in torch.unique(labels):
    cls_mask = labels == cls

    cls_boxes = boxes[cls_mask]
    cls_scores = scores[cls_mask]
    cls_labels = labels[cls_mask]

    keep = nms(cls_boxes, cls_scores, IOU_THRES)

    final_boxes.append(cls_boxes[keep])
    final_scores.append(cls_scores[keep])
    final_labels.append(cls_labels[keep])

if boxes is None or len(final_boxes) == 0:
    print("No objects detected.")
    exit()
boxes = torch.cat(final_boxes, dim=0)
scores = torch.cat(final_scores, dim=0)
labels = torch.cat(final_labels, dim=0)


# ------------------
# DRAW BOXES
# ------------------
TOP_K = 5
scores, idx = scores.sort(descending=True)
idx = idx[:TOP_K]
boxes = boxes[idx]
labels = labels[idx]
scores = scores[idx]


for box, score, label in zip(boxes, scores, labels):
    class_name = CLASS_LABELS.get(label.item(), "Background")
    x1,y1,x2,y2 = box.int().tolist()
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img, f"{class_name}:{score:.2f}",
                (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 1)
    print(f"Detected: {class_name} with confidence {score:.2f}")

cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()