import json
import os

def preprocess_for_my_train_script(coco_json_path, output_path, img_dir):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # 1. Map category IDs to be contiguous (0 to 6)
    categories = data['categories']
    cat_id_map = {cat['id']: i for i, cat in enumerate(categories)}

    # 2. Build a lookup for images
    images_dict = {img['id']: img for img in data['images']}
    
    # 3. Restructure into the format expected by dataloader.py
    processed_data = {}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in images_dict:
            continue
            
        img_info = images_dict[img_id]
        file_name = img_info['file_name']
        
        # Verify image exists
        if not os.path.exists(os.path.join(img_dir, file_name)):
            continue

        if file_name not in processed_data:
            processed_data[file_name] = {"labels": [], "bboxes": []}
        
        processed_data[file_name]["labels"].append(cat_id_map[ann['category_id']])
        processed_data[file_name]["bboxes"].append(ann['bbox'])

    # Save in the format your dataloader.py expects
    with open(output_path, 'w') as f:
        json.dump(processed_data, f)

    print(f"Done! Saved {len(processed_data)} valid images to {output_path}")

# Usage
preprocess_for_my_train_script(r'Object-detection-1\valid\_annotations.coco.json', 'preprocessed_data_valid.json', r'Object-detection-1\valid')