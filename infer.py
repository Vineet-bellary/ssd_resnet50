import torch
import cv2
from torchvision.ops import nms

from model import SSDModel
from anchors import build_all_anchors
from backbone import SimpleSSDBackbone

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 7
NUM_ANCHORS = 6
FEATURE_CHANNELS = [256, 256, 256]

CONF_THRESH = 0.75
IOU_THRESH = 0.5
TOP_K = 10

CLASS_LABELS = {
    1: "vehicles-and-traffic-signals",
    2: "Bike",
    3: "Bus",
    4: "Car",
    5: "Person",
    6: "Traffic Signal",
    7: "Truck",
}

IMAGE_PATH = r"Object-detection-1\train\00044_jpg.rf.702514fdf2d245548ae87cbd091d4ac1.jpg"
MODEL_PATH = "checkpoint_6.pth"

# ---------------- MODEL ----------------
backbone = SimpleSSDBackbone()

model = SSDModel(
    backbone=backbone,
    feature_channels=FEATURE_CHANNELS,
    num_anchors=NUM_ANCHORS,
    num_classes=NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)["model_state"])
model.to(DEVICE)
model.eval()

anchors = build_all_anchors().to(DEVICE)

# ---------------- IMAGE ----------------
img = cv2.imread(IMAGE_PATH)
h, w, _ = img.shape

img_resized = cv2.resize(img, (224, 224))
img_tensor = torch.tensor(img_resized).permute(2,0,1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

# ---------------- INFERENCE ----------------
with torch.no_grad():
    conf_preds, loc_preds = model(img_tensor)

conf_scores = torch.softmax(conf_preds[0], dim=1)
scores, labels = conf_scores.max(dim=1)

mask = (scores > CONF_THRESH) & (labels > 0)

scores = scores[mask]
labels = labels[mask]
loc_preds = loc_preds[0][mask]
anchors = anchors[mask]

# ---------------- DECODE ----------------
ax, ay, aw, ah = anchors[:,0], anchors[:,1], anchors[:,2], anchors[:,3]
tx, ty, tw, th = loc_preds[:,0], loc_preds[:,1], loc_preds[:,2], loc_preds[:,3]

cx = tx * aw + ax
cy = ty * ah + ay
bw = torch.exp(tw) * aw
bh = torch.exp(th) * ah

x1 = cx - bw / 2
y1 = cy - bh / 2
x2 = cx + bw / 2
y2 = cy + bh / 2

boxes = torch.stack([x1, y1, x2, y2], dim=1)
boxes = torch.clamp(boxes, 0, 1)

# scale to original image
boxes[:, [0,2]] *= w
boxes[:, [1,3]] *= h

# ---------------- NMS ----------------
final_boxes = []
final_scores = []
final_labels = []

for cls in torch.unique(labels):
    cls_mask = labels == cls
    keep = nms(boxes[cls_mask], scores[cls_mask], IOU_THRESH)

    final_boxes.append(boxes[cls_mask][keep])
    final_scores.append(scores[cls_mask][keep])
    final_labels.append(labels[cls_mask][keep])

if len(final_boxes) == 0:
    print("No objects detected.")
    exit()

boxes = torch.cat(final_boxes)
scores = torch.cat(final_scores)
labels = torch.cat(final_labels)

# Top-K
scores, idx = scores.sort(descending=True)
idx = idx[:TOP_K]

boxes = boxes[idx]
scores = scores[idx]
labels = labels[idx]

# ---------------- DRAW ----------------
for box, score, label in zip(boxes, scores, labels):
    name = CLASS_LABELS[label.item()]
    x1,y1,x2,y2 = box.int().tolist()

    cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
    cv2.putText(
        img, f"{name}:{score:.2f}",
        (x1, y1-5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
    )

    print(f"Detected {name} with confidence {score:.2f}")

cv2.imshow("SSD Demo Inference", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
