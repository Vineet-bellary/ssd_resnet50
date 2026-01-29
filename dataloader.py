import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from anchors import build_all_anchors
from gt_matching import match_anchors_to_gt

def load_samples(preprocessed_json, image_dir):
    with open(preprocessed_json, "r") as f:
        image_info = json.load(f)

    samples = []
    for file_name, info in image_info.items():
        image_path = rf"{image_dir}/{file_name}"
        labels = info["labels"]
        bboxes = info["bboxes"]
        samples.append((image_path, labels, bboxes))
    return samples

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class DetectionDataset(Dataset):
    def __init__(self, samples, anchors, transform=None):
        self.samples = samples
        self.transform = transform
        # Ensure anchors are on CPU for the matching process
        self.anchors = anchors.cpu() 
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image_path, labels, bboxes = self.samples[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image_tensor = self.transform(image)

        # Matching happens on CPU during data loading
        cls_targets, bbox_targets, labels_mask = match_anchors_to_gt(
            anchors=self.anchors,
            gt_boxes=torch.tensor(bboxes, dtype=torch.float32),
            gt_labels=torch.tensor(labels, dtype=torch.long),
            pos_threshold=0.5,
            neg_threshold=0.4,
            bg_label=0
        )
        return image_tensor, cls_targets, bbox_targets, labels_mask

def ssd_collate_fn(batch):
    # Fixed: Removed the redundant nested function definition
    images, cls_t, bbox_t, masks = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.stack(cls_t, dim=0),
        torch.stack(bbox_t, dim=0),
        torch.stack(masks, dim=0)
    )

# --- Path Configs ---
TRAIN_IMAGE_DIR = r"Object-detection-1\train"
TRAIN_ANNO_PATH = r"preprocessed_data_train.json"

train_samples = load_samples(TRAIN_ANNO_PATH, TRAIN_IMAGE_DIR)

# def main():
#     anchors = build_all_anchors()
    
#     # Fixed: Corrected indentation and variable names
#     train_dataset = DetectionDataset(
#         samples=train_samples,
#         anchors=anchors,
#         transform=transform
#     )
    
#     loader = DataLoader(
#         train_dataset, # Changed 'dataset' to 'train_dataset'
#         batch_size=4,
#         shuffle=True,
#         collate_fn=ssd_collate_fn
#     )
    
#     # Test loop
#     for images, cls_targets, bbox_targets, masks in loader:
#         print(f"Images batch shape: {images.shape}")
#         print(f"CLS Targets shape: {cls_targets.shape}")
#         break
    
#     return loader

# if __name__ == "__main__":
#     main()