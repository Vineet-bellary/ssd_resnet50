import torch
import math

def cxcywh_to_xyminmax(boxes):
    '''
    Convert boxes from (cx, cy, w, h) to (x_min, y_min, x_max, y_max)
    
    :param boxes: boxes is a tensor of shape (num_boxes, 4)
    i.e. each row is (cx, cy, w, h)
    '''
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2
    
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)

def compute_iou(boxes1, boxes2):
    '''
    Compute IoU between two sets of boxes.
    
    :param boxes1: tensor of shape (N, 4) in (x_min, y_min, x_max, y_max)
    :param boxes2: tensor of shape (M, 4) in (x_min, y_min, x_max, y_max)
    :return: IoU matrix of shape (N, M)
    '''
    b1 = boxes1[:, None, :] # (N, 1, 4)
    b2 = boxes2[None, :, :] # (1, M, 4)
    
    # Intersection
    inter_xmin = torch.max(b1[..., 0], b2[..., 0])
    inter_ymin = torch.max(b1[..., 1], b2[..., 1])
    inter_xmax = torch.min(b1[..., 2], b2[..., 2])
    inter_ymax = torch.min(b1[..., 3], b2[..., 3])

    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
    inter_area = inter_w * inter_h
    
    # Areas
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    # Compute IoU
    union_area = area1 + area2 - inter_area
    
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    return iou

def force_match_gt(iou_matrix):
    """
    iou_matrix: Tensor [A, G]

    returns:
        best_anchor_for_gt: Tensor [G]
        (index of anchor with max IoU for each GT)
    """
    # For each GT (column), find anchor with max IoU
    best_anchor_for_gt = torch.argmax(iou_matrix, dim=0)
    return best_anchor_for_gt

def classify_anchors(iou_matrix, best_anchor_for_gt, pos_threshold=0.5, neg_threshold=0.4):
    """
    iou_matrix: Tensor [A, G]
    best_anchor_for_gt: Tensor [G]

    returns:
        labels: Tensor [A] with values
                1 = positive
                0 = negative
               -1 = ignore
        matched_gt_idx: Tensor [A] (GT index for positives, undefined otherwise)
    """
    A, G = iou_matrix.shape
    
    labels = -1 *torch.ones(A, dtype=torch.int64)
    matched_gt_idx = torch.full((A,), -1, dtype=torch.long)
    
    # For each anchor, best GT & IoU
    max_iou_per_anchor, best_gt_idx = torch.max(iou_matrix, dim=1)
    
    # STEP 3: forced positives
    labels[best_anchor_for_gt] = 1
    matched_gt_idx[best_anchor_for_gt] = torch.arange(G)

    # STEP 4: threshold-based rules (skip forced positives)
    for a in range(A):
        if labels[a] == 1:
            continue

        if max_iou_per_anchor[a] >= pos_threshold:
            labels[a] = 1
            matched_gt_idx[a] = best_gt_idx[a]
        elif max_iou_per_anchor[a] < neg_threshold:
            labels[a] = 0  # negative
        else:
            labels[a] = -1  # ignore

    return labels, matched_gt_idx

def compute_bbox_targets(anchors, gt_boxes, matched_gt_idx, labels):
    """
    anchors: Tensor [A, 4] in [cx, cy, w, h]
    gt_boxes: Tensor [G, 4] in [cx, cy, w, h]
    matched_gt_idx: Tensor [A] (GT index for positives, -1 otherwise)
    labels: Tensor [A] (1=pos, 0=neg, -1=ignore)

    returns:
        bbox_targets: Tensor [A, 4]
    """
    A = anchors.shape[0]
    bbox_targets = torch.zeros((A, 4), dtype=torch.float32, device=anchors.device)

    pos_mask = labels == 1
    pos_indices = torch.where(pos_mask)[0]

    if len(pos_indices) == 0:
        return bbox_targets  # no positives in this image

    a = anchors[pos_indices]
    g = gt_boxes[matched_gt_idx[pos_indices]]

    ax, ay, aw, ah = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    gx, gy, gw, gh = g[:, 0], g[:, 1], g[:, 2], g[:, 3]

    tx = (gx - ax) / aw
    ty = (gy - ay) / ah
    tw = torch.log(gw / aw)
    th = torch.log(gh / ah)

    bbox_targets[pos_indices] = torch.stack([tx, ty, tw, th], dim=1)
    return bbox_targets

def match_anchors_to_gt(
    anchors,
    gt_boxes,
    gt_labels,
    pos_threshold=0.5,
    neg_threshold=0.4,
    bg_label=0
):
    device = anchors.device
    assert anchors.is_cuda, "ANCHORS NOT ON CUDA"
    # print("ANCHORS DEVICE:", anchors.device)
    # print("GT BOXES DEVICE:", gt_boxes.device)  

    gt_boxes = gt_boxes.to(device)
    gt_labels = gt_labels.to(device)

    """
    anchors: Tensor [A, 4] (cx,cy,w,h)
    gt_boxes: Tensor [G, 4] (cx,cy,w,h)
    gt_labels: Tensor [G]

    returns:
        cls_targets: Tensor [A]
        bbox_targets: Tensor [A, 4]
        labels_mask: Tensor [A] (1=pos, 0=neg, -1=ignore)
    """

    # STEP 1: convert to corners
    anchors_xyxy = cxcywh_to_xyminmax(anchors)
    gt_xyxy = cxcywh_to_xyminmax(gt_boxes)

    # STEP 2: IoU matrix
    iou_matrix = compute_iou(anchors_xyxy, gt_xyxy)

    # STEP 3: force one anchor per GT
    best_anchor_for_gt = force_match_gt(iou_matrix)

    # STEP 4: classify anchors
    labels_mask, matched_gt_idx = classify_anchors(
        iou_matrix,
        best_anchor_for_gt,
        pos_threshold,
        neg_threshold
    )

    # STEP 5: bbox regression targets
    bbox_targets = compute_bbox_targets(
        anchors,
        gt_boxes,
        matched_gt_idx,
        labels_mask
    )

    # STEP 6: class targets
    A = anchors.shape[0]
    cls_targets = torch.full((A,), bg_label, dtype=torch.long, device=anchors.device)

    pos_mask = labels_mask == 1
    ignore_mask = labels_mask == -1

    cls_targets[pos_mask] = gt_labels[matched_gt_idx[pos_mask]]
    cls_targets[ignore_mask] = -1  # ignored in loss

    return cls_targets, bbox_targets, labels_mask

match_anchors_to_gt