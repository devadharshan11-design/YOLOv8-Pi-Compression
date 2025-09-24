from ultralytics import YOLO

def main():
    # --- 1. Use the FULL, ABSOLUTE path to the last checkpoint ---
    # This tells the script exactly where to find the file on your D: drive.
    model = YOLO('D:/PROJECTS/edge_ai_compression/runs/detect/yolov8n_pruned_finetuned/weights/last.pt')
    
    print("Resuming fine-tuning from the last checkpoint...")
    
    # --- 2. Call train() with the resume=True argument ---
    # The trainer will automatically continue from where it left off.
    model.train(resume=True)

if __name__ == '__main__':
    main()