import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv

def main():
    # --- 1. Load the FULL, ORIGINAL TRAINED model (with its weights) ---
    model_path = 'C:/Users/devad/runs/detect/yolov8n_baseline_student/weights/best.pt'
    model = YOLO(model_path)
    pytorch_model = model.model
    
    print("Original model loaded. Starting surgical pruning...")
    
    # --- 2. Apply the successful pruning logic directly to the loaded model ---
    for module in pytorch_model.modules():
        if isinstance(module, C2f):
            if len(module.m) > 1: # Only prune if there's more than one bottleneck
                old_cv2 = module.cv2
                module.m = module.m[:-1]
                
                # Dynamically calculate channels
                channels_from_cv1 = module.cv1.conv.out_channels
                channels_from_bottlenecks = len(module.m) * module.m[0].cv2.conv.out_channels
                new_in_channels = channels_from_cv1 + channels_from_bottlenecks
                
                module.cv2 = Conv(
                    new_in_channels,
                    old_cv2.conv.out_channels,
                    old_cv2.conv.kernel_size,
                    old_cv2.conv.stride,
                    act=old_cv2.act
                )
    
    print("Model successfully pruned in memory. Starting fine-tuning...")

    # --- 3. Fine-tune the now-pruned model ---
    # No need to load weights; they are already part of the modified model object.
    model.train(
        data='D:/PROJECTS/edge_ai_compression_local/coco.yaml',
        epochs=50,
        batch=16,
        imgsz=640,
        project='runs/detect',
        name='yolov8n_pruned_finetuned',
        # Added to ensure a clean run
        exist_ok=True 
    )

if __name__ == '__main__':
    main()