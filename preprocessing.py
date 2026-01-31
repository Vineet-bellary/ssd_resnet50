import json
import os

def json_preprocessor(coco_json_path, output_path, img_dir):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # -------------------------------------------------
    # 1. Category ID mapping
    #    Background = 0
    #    Object classes start from 1
    # -------------------------------------------------
    VALID_CLASSES = ["knife", "pistol", "rifle", "shotgun"]

    cat_id_map = {}
    new_id = 1

    for cat in data["categories"]:
        if cat["name"] in VALID_CLASSES:
            cat_id_map[cat["id"]] = new_id
            new_id += 1

    # -------------------------------------------------
    # 2. Image lookup by image_id
    # -------------------------------------------------
    images_dict = {img["id"]: img for img in data["images"]}

    processed_data = {}

    # -------------------------------------------------
    # 3. Process annotations
    # -------------------------------------------------
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in images_dict:
            continue

        img_info = images_dict[img_id]
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        image_path = os.path.join(img_dir, file_name)
        if not os.path.exists(image_path):
            continue

        # COCO bbox: [x_min, y_min, width, height] (pixels)
        x, y, w, h = ann["bbox"]

        # Skip invalid boxes
        if w <= 0 or h <= 0:
            continue

        # -------------------------------------------------
        # Convert to cx, cy, w, h (normalized)
        # -------------------------------------------------
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        # Skip boxes outside image after normalization
        if not (0 < cx < 1 and 0 < cy < 1 and 0 < nw <= 1 and 0 < nh <= 1):
            continue

        if file_name not in processed_data:
            processed_data[file_name] = {
                "labels": [],
                "bboxes": []
            }

        # Label remap (background = 0, objects start from 1)
# Skip annotations whose category is not in VALID_CLASSES
        if ann["category_id"] not in cat_id_map:
            continue
        new_label = cat_id_map[ann["category_id"]]

        processed_data[file_name]["labels"].append(new_label)
        processed_data[file_name]["bboxes"].append([cx, cy, nw, nh])

    # -------------------------------------------------
    # 4. Save processed JSON
    # -------------------------------------------------
    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=2)

    print(f"[OK] Saved {len(processed_data)} images to {output_path}")


# -------------------------------------------------
# USAGE
# -------------------------------------------------

json_preprocessor(
    r"ssd-object-detection-5\train\_annotations.coco.json",
    "preprocessed_weapon_train.json",
    r"ssd-object-detection-5\train"
)

json_preprocessor(
    r"ssd-object-detection-5\valid\_annotations.coco.json",
    "preprocessed_weapon_valid.json",
    r"ssd-object-detection-5\valid"
)
