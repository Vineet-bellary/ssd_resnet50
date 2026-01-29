import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
from model import SSDModel
from new_backbone import ResNet50Backbone
from anchors import build_all_anchors
from gt_matching import decode_boxes  # Ensure this uses the 0.1/0.2 variances!

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = r"models\checkpoint_3.pth"
NUM_CLASSES = 7
CONF_THRESHOLD = 0.2  # Minimum score to show a box
IOU_THRESHOLD = 0.6  # NMS threshold


def run_inference(image_path, model, anchors, class_names):
    # 1. Preprocess
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    h_orig, w_orig, _ = orig_img.shape
    img_resized = cv2.resize(orig_img, (224, 224))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    # 2. Forward Pass
    model.eval()
    with torch.no_grad():
        cls_logits, bbox_preds = model(img_tensor)

    # 3. Process Scores
    cls_probs = torch.softmax(cls_logits[0], dim=-1)
    scores, labels = torch.max(cls_probs, dim=-1)

    # --- DEBUG: PRINT RAW OUTPUT ---
    top_scores, top_indices = torch.topk(scores, 10)
    top_labels = labels[top_indices]
    
    print("\n" + "="*40)
    print("DEBUG: TOP 10 RAW PREDICTIONS")
    print("="*40)
    for i in range(10):
        s = top_scores[i].item()
        l = top_labels[i].item()
        name = class_map.get(l, f"Unknown({l})")
        print(f"Rank {i+1}: ID {l} ({name:12}) | Score: {s:.4f}")
    print("="*40 + "\n")

    # 4. Multi-Object Filtering
    mask = (labels > 0) & (scores > CONF_THRESHOLD)

    if not mask.any():
        print("\n[Terminal Log] No objects detected above threshold.")
        return orig_img

    # This keeps ALL objects that passed the mask
    filtered_preds = bbox_preds[0][mask]
    filtered_anchors = anchors[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    decoded_boxes = decode_boxes(filtered_preds, filtered_anchors)

    # 5. NMS (Removes duplicates, but keeps different objects)
    keep_idx = torchvision.ops.nms(decoded_boxes, filtered_scores, IOU_THRESHOLD)

    final_boxes = decoded_boxes[keep_idx]
    final_scores = filtered_scores[keep_idx]
    final_labels = filtered_labels[keep_idx]

    # 6. Terminal Logging & Visualization Loop
    print(f"\n--- Detections in {image_path.split('/')[-1]} ---")
    for i in range(len(final_boxes)):
        box = final_boxes[i].cpu().numpy()
        cls_id = final_labels[i].item()
        conf = final_scores[i].item()
        name = class_names.get(cls_id, f"Unknown({cls_id})")

        # Scaled Coordinates
        xmin = int(box[0] * w_orig)
        ymin = int(box[1] * h_orig)
        xmax = int(box[2] * w_orig)
        ymax = int(box[3] * h_orig)

        # --- TERMINAL LOG ---
        print(
            f"[{i+1}] {name:12} | Conf: {conf:.4f} | Box: [{xmin}, {ymin}, {xmax}, {ymax}]"
        )

        # Visualization
        cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label_text = f"{name}: {conf:.2f}"
        cv2.putText(
            orig_img,
            label_text,
            (xmin, max(20, ymin - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    print(f"Total objects found: {len(final_boxes)}")
    print("-------------------------------------------\n")
    return orig_img


if __name__ == "__main__":
    # Setup
    test_img_path = r"Object-detection-1\test\40_jpeg.rf.96c7a1c0150a9ecfc683b194fc6012f7.jpg"

    class_map = {
        1: "vehicles-and-traffic-signals",
        2: "bike",
        3: "bus",
        4: "car",
        5: "person",
        6: "motorcycle",
        7: "truck",
    }

    anchors = build_all_anchors().to(DEVICE)

    model = SSDModel(ResNet50Backbone(), [512, 1024, 2048], 6, NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    result = run_inference(test_img_path, model, anchors, class_map)
    # cv2.imwrite("output.jpg", result)
    # Show the image in a window
    cv2.imshow("SSD Detection Result", result)

    # Wait for any key press
    print("Click on the image window and press any key to close...")
    cv2.waitKey(0)

    # Clean up and close the window
    cv2.destroyAllWindows()
