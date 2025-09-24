import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv

def main():
    model_path = 'C:/Users/devad/runs/detect/yolov8n_baseline_student/weights/best.pt'
    model = YOLO(model_path)
    pytorch_model = model.model
    
    print("Starting surgical pruning to create weights...")
    
    for module in pytorch_model.modules():
        if isinstance(module, C2f):
            if len(module.m) > 1:
                old_cv2 = module.cv2
                module.m = module.m[:-1]
                
                # Dynamically calculate channels
                channels_from_cv1 = module.cv1.conv.out_channels
                channels_from_bottlenecks = len(module.m) * module.m[0].cv2.conv.out_channels
                new_in_channels = channels_from_cv1 + channels_from_bottlenecks
                
                module.cv2 = Conv(new_in_channels, old_cv2.conv.out_channels,
                                  old_cv2.conv.kernel_size, old_cv2.conv.stride, act=old_cv2.act)
    
    pruned_model_path = 'yolov8n_pruned.pt'
    torch.save(pytorch_model.state_dict(), pruned_model_path)
    print(f"Pruned model weights saved to '{pruned_model_path}'")

if __name__ == '__main__':
    main()