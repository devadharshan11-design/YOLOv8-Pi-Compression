# Edge-AI Compression of YOLOv8 for Raspberry Pi

## Summary
This project demonstrates a pipeline for compressing a YOLOv8n object detection model to run efficiently on a Raspberry Pi 5. The process involves structured pruning, fine-tuning to recover accuracy, and post-training INT8 quantization.

## Key Features
- **Structured Pruning:** Surgically removed redundant layers from the YOLOv8n architecture, reducing parameters by over 27%.
- **Fine-Tuning:** Retrained the pruned model to recover performance to a level nearly identical to the original.
- **Quantization:** Converted the final model to INT8 for faster inference on edge devices.

## Final Results
The final optimized model achieves a 34% speed improvement with a negligible 0.3% drop in accuracy compared to the baseline.

| Model Version | Parameters | Accuracy (mAP50-95) | Speed (Pi 5 FPS) |
| :--- | :--- | :--- | :--- |
| **Baseline YOLOv8n** | 3.16M | 0.355 | 3.08 |
| **Optimized Model**| 3.05M | 0.354 | 4.13 |

## How to Use
1.  Set up the environment: `pip install -r requirements.txt`
2.  The `finetune.py` script contains the full pipeline for loading the original model, pruning it, and fine-tuning it.
3.  The `benchmark.py` script can be used to test `.tflite` models on a Raspberry Pi.