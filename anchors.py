import torch
import math

def generate_anchors(feature_map_size, scales, aspect_ratios):
    
    # setup
    # scales = [0.2]
    # aspect_ratios = [1, 2, 0.5]
    anchors = []
    
    for i in range(feature_map_size):
        for j in range(feature_map_size):
            cx = (j + 0.5) / feature_map_size
            cy = (i + 0.5) / feature_map_size
            
            for s in scales:
                for ar in aspect_ratios:
                    w = s * math.sqrt(ar)
                    h = s / math.sqrt(ar)
                    
                    anchors.append([cx, cy, w, h])
    
    anchors = torch.tensor(anchors, dtype=torch.float32)
    return anchors

def clip_anchors(anchors):
    
    '''
    Docstring for clip_anchors
    
    :param anchors: anchors is a tensor of shape (num_anchors, 4)
    i.e. each row is (cx, cy, w, h)
    '''
    cx, cy, w, h = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    # print(h[:3])
    # Here we convert (cx, cy, w, h) to (x_min, y_min, x_max, y_max)
    x_min = cx - w / 2
    y_min = cy - h / 2
    x_max = cx + w / 2
    y_max = cy + h / 2

    x_min = torch.clamp(x_min, 0.0, 1.0)
    y_min = torch.clamp(y_min, 0.0, 1.0)
    x_max = torch.clamp(x_max, 0.0, 1.0)
    y_max = torch.clamp(y_max, 0.0, 1.0)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    return torch.stack([cx, cy, w, h], dim=1)

def build_all_anchors():
    # Configurations
    feature_maps = [28, 14, 7]
    scales = {
        28: [0.25, 0.35],
        14: [0.50, 0.60],
        7: [0.80, 0.90]
    }
    aspect_ratios = [1, 2, 0.5]
    all_anchors = []
    
    # Generate anchors for each feature map
    for fmap_size in feature_maps:
        anchors = generate_anchors(
            feature_map_size=fmap_size,
            scales=scales[fmap_size],
            aspect_ratios=aspect_ratios
        )
        all_anchors.append(anchors)
    
    all_anchors = torch.cat(all_anchors, dim=0)
    all_anchors = clip_anchors(all_anchors)
    
    return all_anchors
