import torch
from torchvision.ops import nms
from anchors import build_all_anchors

def infer_one_image(image_tensor, model, device, conf_thresh=0.6, iou_thresh=0.7, top_k=50):
    model.eval()

    image_tensor = image_tensor.unsqueeze(0).to(device)

    anchors = build_all_anchors().to(device)

    with torch.no_grad():
        conf_preds, loc_preds = model(image_tensor)

    conf_scores = torch.softmax(conf_preds[0], dim=1)
    scores, labels = conf_scores.max(dim=1)

    mask = (scores > conf_thresh) & (labels > 0)

    if mask.sum() == 0:
        return (
            torch.empty((0,4)),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,))
        )

    scores = scores[mask]
    labels = labels[mask]
    loc_preds = loc_preds[0][mask]
    anchors = anchors[mask]

    ax, ay, aw, ah = anchors[:,0], anchors[:,1], anchors[:,2], anchors[:,3]
    tx, ty, tw, th = loc_preds[:,0], loc_preds[:,1], loc_preds[:,2], loc_preds[:,3]

    cx = tx * aw + ax
    cy = ty * ah + ay
    w  = torch.exp(tw) * aw
    h  = torch.exp(th) * ah

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    boxes = torch.clamp(boxes, 0, 1)

    final_boxes = []
    final_scores = []
    final_labels = []

    for cls in torch.unique(labels):
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_labels = labels[cls_mask]

        keep = nms(cls_boxes, cls_scores, iou_thresh)

        final_boxes.append(cls_boxes[keep])
        final_scores.append(cls_scores[keep])
        final_labels.append(cls_labels[keep])

    if len(final_boxes) == 0:
        return (
            torch.empty((0,4)),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,))
        )

    boxes = torch.cat(final_boxes, dim=0)
    scores = torch.cat(final_scores, dim=0)
    labels = torch.cat(final_labels, dim=0)

    scores, idx = scores.sort(descending=True)
    idx = idx[:top_k]

    return boxes[idx], labels[idx], scores[idx]
