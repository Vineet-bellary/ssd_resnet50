import json

# Loading Data
with open(r"Object-detection-1\train\_annotations.coco.json", "r") as f:
    data = json.load(f)
# print(data.keys())
print("Data Loaded Successfully!!!\n\n")

"""
We need a tuple with Images, Labels, bboxes to feed into Model.
Images:
Labels: 
bboxes: 
"""

# Classes
categories = data["categories"]
classes = []

for cat in categories:
    classes.append(cat["name"])
# print(f"{classes}\n\n")

# Metadata Building
images = data["images"]

imgs = {}
for image in images:
    imgs[image["id"]] = image["file_name"]

annotations = data["annotations"]

image_info = {}
for ann in annotations:
    img_id = ann["image_id"]
    file_name = imgs[img_id]
    bbox = ann["bbox"]
    label = ann["category_id"]

    if file_name not in image_info:
        image_info[file_name] = {"labels": [], "bboxes": []}

    image_info[file_name]["labels"].append(label)
    image_info[file_name]["bboxes"].append(bbox)

"""
Upto here we have made a metadata from the Coco annotation file

Next step is to make the data into model ready format 
"""


# For normalizing BBoxes
for file_name, info in image_info.items():
    for i, bbox in enumerate(info["bboxes"]):
        # fetching bbox values
        x_min = bbox[0]
        y_min = bbox[1]
        width = bbox[2]
        height = bbox[3]

        # Calculating center coordinates
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        # Normalizing
        x_center /= 224
        y_center /= 224
        width /= 224
        height /= 224

        # Updating bbox
        info["bboxes"][i] = [x_center, y_center, width, height]

# for k, v in list(image_info.items())[:2]:
#     print(f"Image: {k} -> Labels: {v['labels']} -> BBoxes: {v['bboxes']}\n")

# for f, info in list(image_info.items())[:3]:
#     print(f"Image: {f} -> Labels: {info['labels']} -> BBoxes: {info['bboxes']}\n")

cat_id_to_label = {}
label_id_to_cat = {}
for idx, cat in enumerate(categories):
    cat_id_to_label[cat["id"]] = idx
    label_id_to_cat[idx] = cat["id"]

print(f"Category ID to Label Mapping: {cat_id_to_label}\n")
print(f"Label ID to Category Mapping: {label_id_to_cat}\n")


for file_name, info in image_info.items():
    new_labels = []
    for cat_id in info["labels"]:
        new_labels.append(cat_id_to_label[cat_id])
    info["labels"] = new_labels

# for k, v in list(image_info.items())[:5]:
#     print(k, v["labels"])

with open("preprocessed_data.json", "w") as f:
    json.dump(image_info, f)
    
print("Preprocessed data saved to 'preprocessed_data.json'")