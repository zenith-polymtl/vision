from ultralytics import YOLO

# 1. Load your original PyTorch weights
# Ensure best.pt is in the current directory
model = YOLO('best_nano.pt') 

# 2. Export to TensorRT format
# format='engine' will handle the .pt -> .onnx -> .engine path automatically
# We specify half=True for FP16 and imgsz=640 (standard for YOLO nano)
path = model.export(
    format='engine', 
    device=0, 
    half=True, 
    imgsz=640,
    simplify=True  # This cleans up the ONNX graph before TRT sees it
)

print(f"Masterpiece saved at: {path}")