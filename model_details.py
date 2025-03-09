import logging
from tabulate import tabulate
from ultralytics import YOLO

# Disable YOLO logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)

model = YOLO("yolov8n.pt")

class_names = [(idx+1, name) for idx, name in model.names.items()]
print(tabulate(class_names, headers=["Class ID", "Class Name"], tablefmt="grid"))

print("Model Summary: ")
model.info()

print("Model Config: ")
print(model.yaml)

print("Model parameters:")
print(model.model)