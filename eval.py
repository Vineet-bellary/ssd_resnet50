import torch

from model import SSDModel
from backbone import SimpleSSDBackbone
from dataloader import DetectionDataset, samples, transform
from infer_utils import infer_one_image
from gt_matching import compute_iou

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 7
FEATURE_CHANNELS = [256, 256, 256]
NUM_ANCHORS = 6

# Load model
backbone = SimpleSSDBackbone()

model = SSDModel(
    backbone=backbone,
    feature_channels=FEATURE_CHANNELS,
    num_anchors=NUM_ANCHORS,
    num_classes=NUM_CLASSES
)

MODEL_PATH = r"models\checkpoint_bs64_33.pth"

'''
When use checkpoint as model added 'model_state' key
When using direct saved model, no 'model_state' key
'''
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)['model_state'])
model.to(DEVICE)
model.eval()

dataset = DetectionDataset(samples, transform=transform)

def evaluate_image(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.5):
    TP = FP = FN = 0
    matched_gt = set()

    if len(gt_boxes) == 0:
        FP = len(pred_boxes)
        return TP, FP, FN

    if len(pred_boxes) == 0:
        FN = len(gt_boxes)
        return TP, FP, FN

    iou_matrix = compute_iou(pred_boxes, gt_boxes)
    print("GT boxes:", gt_boxes[:2])
    print("Pred boxes:", pred_boxes[:2])
    for p in range(len(pred_boxes)):
        best_iou, best_gt = torch.max(iou_matrix[p], dim=0)

        if best_iou >= iou_thresh:
            if best_gt.item() not in matched_gt and pred_labels[p] == gt_labels[best_gt]:
                TP += 1
                matched_gt.add(best_gt.item())
            else:
                FP += 1
        else:
            FP += 1

    FN = len(gt_boxes) - len(matched_gt)
    return TP, FP, FN


total_TP = total_FP = total_FN = 0

for i in range(len(dataset)):
    image, gt_labels, gt_boxes = dataset[i]
    
    gt_boxes = gt_boxes.to(DEVICE)
    gt_labels = gt_labels.to(DEVICE)

    pred_boxes, pred_labels, pred_scores = infer_one_image(
        image_tensor=image,
        model=model,
        device=DEVICE
    )
    
    pred_boxes = pred_boxes.to(DEVICE)
    pred_labels = pred_labels.to(DEVICE)

    TP, FP, FN = evaluate_image(pred_boxes, pred_labels, gt_boxes, gt_labels)

    total_TP += TP
    total_FP += FP
    total_FN += FN

precision = total_TP / (total_TP + total_FP + 1e-6)
recall = total_TP / (total_TP + total_FN + 1e-6)

print("====== RESULTS ======")
print("TP:", total_TP)
print("FP:", total_FP)
print("FN:", total_FN)
print("Precision:", precision)
print("Recall:", recall)
