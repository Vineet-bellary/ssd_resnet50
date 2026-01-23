from roboflow import Roboflow

rf = Roboflow(api_key="hRrqiECkoEkefllT6CWH")
project = rf.workspace("small-object-detections-smart-surveillance-system").project(
    "object-detection-axukj-li4tq"
)
version = project.version(1)
dataset = version.download("coco")
