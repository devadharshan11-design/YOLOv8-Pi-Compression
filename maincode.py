import torch
from ultralytics import YOLO
import os

def main():
    BASE = r"D:\PROJECTS\edge_ai_compression_local"
    YAML_PATH = os.path.join(BASE, "coco.yaml")
    
    RUN_NAME = "yolov8m_teacher_run_low_lr" 
    
    # Load the 'last.pt' file to resume a completed run
    START_MODEL = os.path.join(BASE, RUN_NAME, "weights", "last.pt")

    if not os.path.isfile(START_MODEL):
        print("last.pt not found, starting from pretrained yolov8m.pt")
        START_MODEL = "yolov8m.pt"

    model = YOLO(START_MODEL)

    model.train(
        data=YAML_PATH,
        epochs=100, # Increase the number of epochs to train for
        optimizer='SGD',
        lr0=0.001,
        imgsz=640,
        batch=8,
        workers=3,
        device=0 if torch.cuda.is_available() else "cpu",
        project=BASE,
        name=RUN_NAME,
        exist_ok=True,
        patience=20,
        resume=True # Add this line to resume training
    )

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()