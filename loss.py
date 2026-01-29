import torch
import torch.nn.functional as F

def localization_loss(bbox_preds, bbox_targets, labels_mask):
    """
    bbox_preds: Tensor [N, A, 4]
    bbox_targets: Tensor [N, A, 4]
    labels_mask: Tensor [N, A] (1=pos, 0=neg, -1=ignore)

    returns:
        loc_loss: scalar Tensor
    """
    
    pos_mask = (labels_mask == 1).to(labels_mask.device)
    num_pos = pos_mask.sum().clamp(min=1).float()
    
    # Selecting only positive anchors
    pred = bbox_preds[pos_mask]
    target = bbox_targets[pos_mask]
    
    loss = F.smooth_l1_loss(pred, target, reduction="sum")
    
    return loss / num_pos

def confidence_loss_basic(cls_logits, cls_targets):
    """
    cls_logits: Tensor [A, C+1]
    cls_targets: Tensor [A] (0=bg, 1..C=object, -1=ignore)

    returns:
        conf_loss_per_anchor: Tensor [A]
    """

    valid_mask = (cls_targets != -1).to(cls_targets.device)  # exclude ignored
    logits = cls_logits[valid_mask]
    targets = cls_targets[valid_mask]

    # Cross entropy per anchor (no reduction)
    loss = F.cross_entropy(
        logits,
        targets,
        reduction="none"
    )

    return loss, valid_mask

def hard_negative_mining(conf_loss, cls_targets, labels_mask, neg_pos_ratio=3):
    """
    conf_loss: Tensor [N] (per-anchor loss, from valid anchors only)
    cls_targets: Tensor [A]
    labels_mask: Tensor [A]
    """

    device = labels_mask.device
    
    pos_mask = (labels_mask == 1).to(device)
    neg_mask = (labels_mask == 0).to(device)

    num_pos = pos_mask.sum().item()
    num_neg = neg_mask.sum().item()

    if num_pos == 0:
        return pos_mask  # edge case

    max_neg = min(neg_pos_ratio * num_pos, num_neg)

    # Loss only for negatives
    neg_losses = conf_loss.clone()
    neg_losses[~neg_mask] = -1  # exclude positives & ignores

    # Sort negatives by loss
    _, idx = torch.sort(neg_losses, descending=True)

    hard_neg_mask = torch.zeros_like(labels_mask, dtype=torch.bool)
    hard_neg_mask[idx[:max_neg]] = True

    final_mask = pos_mask | hard_neg_mask
    return final_mask


def ssd_loss(
    cls_logits,
    bbox_preds,
    cls_targets,
    bbox_targets,
    labels_mask,
    neg_pos_ratio=3
):
    """
    cls_logits: Tensor [A, C+1]
    bbox_preds: Tensor [A, 4]
    cls_targets: Tensor [A]
    bbox_targets: Tensor [A, 4]
    labels_mask: Tensor [A]
    """
    
    # Flatten the batch dimension so the existing logic works on all anchors at once
    B, A, C_plus_1 = cls_logits.shape
    
    cls_logits = cls_logits.view(-1, C_plus_1) # [B*A, C+1]
    bbox_preds = bbox_preds.view(-1, 4)        # [B*A, 4]
    cls_targets = cls_targets.view(-1)         # [B*A]
    bbox_targets = bbox_targets.view(-1, 4)    # [B*A, 4]
    labels_mask = labels_mask.view(-1)         # [B*A]

    # Now the rest of your functions (localization_loss, hard_negative_mining) 
    # will work perfectly because they treat the flattened batch as one large image.

    # -------------------------
    # 1) Localization loss
    # -------------------------
    loc_loss = localization_loss(
        bbox_preds,
        bbox_targets,
        labels_mask
    )

    # -------------------------
    # 2) Confidence loss (per anchor)
    # -------------------------
    conf_loss_per_anchor, valid_mask = confidence_loss_basic(
        cls_logits,
        cls_targets
    )
    
    # -------------------------
    # 3) Hard negative mining
    # -------------------------
    conf_mask = hard_negative_mining(
        conf_loss_per_anchor, 
        cls_targets[valid_mask], 
        labels_mask[valid_mask], 
        neg_pos_ratio
    )

    # -------------------------
    # 4) Final confidence loss
    # -------------------------
    conf_loss = conf_loss_per_anchor[conf_mask].sum()

    num_pos = (labels_mask == 1).sum().clamp(min=1).float()
    conf_loss = conf_loss / num_pos

    # -------------------------
    # 5) Total loss
    # -------------------------
    total_loss = loc_loss + conf_loss

    return total_loss, loc_loss, conf_loss
