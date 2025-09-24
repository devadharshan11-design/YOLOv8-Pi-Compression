import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Conv

def main():
    # --- 1. Load the FULL model structure from the original YAML file ---
    model = YOLO('yolov8n.yaml')
    pytorch_model = model.model
    
    print("Applying surgical pruning to the fresh model structure...")
    
    # --- 2. Apply the EXACT SAME pruning logic to the fresh structure ---
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

    print("Pruning of structure complete.")

    # --- 3. Now, load the pruned weights into the perfectly matching pruned structure ---
    pruned_weights_path = 'yolov8n_pruned.pt'
    state_dict = torch.load(pruned_weights_path)
    model.model.load_state_dict(state_dict)
    
    print("Successfully loaded pruned weights.")
    
    # --- 4. Run validation ---
    print("Starting validation...")
    results = model.val(data='D:/PROJECTS/edge_ai_compression_local/coco.yaml')
    print(results)

if __name__ == '__main__':
    main()