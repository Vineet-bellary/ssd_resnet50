import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Configurations
IMAGE_DIR = r"Object-detection-1\train"

with open("preprocessed_data.json", "r") as f:
    image_info = json.load(f)

# Sample creation
samples = []

for file_name, info in image_info.items():
    image_path = rf"{IMAGE_DIR}\{file_name}"
    labels = info["labels"]
    bboxes = info["bboxes"]

    samples.append((image_path, labels, bboxes))

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset class
class DetectionDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image_path, labels, bboxes = self.samples[index]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Transforms
        if self.transform is not None:
            image_tensor = self.transform(image)

        # Tensor conversion
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        
        return image_tensor, labels_tensor, bboxes_tensor

def ssd_collate_fn(batch):
    """
    batch = list of samples from Dataset
    each sample = (image_tensor, labels_tensor, bboxes_tensor)
    """
    images = []
    labels = []
    bboxes = []
    
    for sample in batch:
        img, lab, bb = sample
        images.append(img)
        labels.append(lab)
        bboxes.append(bb)
        
    images = torch.stack(images, dim=0)
    
    return images, labels, bboxes


def main():
    dataset = DetectionDataset(samples, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size = 4,
        shuffle=True,
        collate_fn=ssd_collate_fn
    )
    
    for images, labels, bboxes in loader:
        print(images.shape)
        print(type(labels), len(labels))
        print(type(bboxes), len(bboxes))
        print(labels[0].shape, bboxes[0].shape)
        break
    
    return loader

if __name__ == "__main__":
    main()