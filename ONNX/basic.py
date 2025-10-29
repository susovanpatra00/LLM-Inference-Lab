import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import time
import netron


# ============================================================================
# 1. Define the Model
# ============================================================================
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 32 * 32, 10)
    
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================================
# 2. Create Model and Test Input
# ============================================================================
model = SmallCNN().eval()
x = torch.randn(1, 3, 32, 32)
torch_out = model(x)
print("PyTorch output shape:", torch_out.shape)


# ============================================================================
# 3. Export to ONNX
# ============================================================================
torch.onnx.export(
    model,                          # model being run
    x,                              # example input
    "smallcnn.onnx",                # where to save
    export_params=True,             # store trained weights
    opset_version=17,               # ONNX opset
    do_constant_folding=True,       # fold constants
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch'}, 
        'output': {0: 'batch'}
    }
)
print("✅ Model exported to smallcnn.onnx")


# ============================================================================
# 4. Visualize with Netron (optional)
# ============================================================================
# Uncomment to visualize in browser:
# netron.start("smallcnn.onnx")


# ============================================================================
# 5. Load ONNX Model and Run Inference
# ============================================================================
session = ort.InferenceSession(
    "smallcnn.onnx", 
    providers=["CPUExecutionProvider"]
)

# Prepare input
inputs = {session.get_inputs()[0].name: x.numpy()}

# Run inference
onnx_out = session.run(None, inputs)[0]

print("\n" + "="*60)
print("VALIDATION")
print("="*60)
print("ONNX output shape:", onnx_out.shape)
print("Diff from PyTorch:", np.max(np.abs(onnx_out - torch_out.detach().numpy())))


# ============================================================================
# 6. Performance Benchmark
# ============================================================================
def time_it(func, n=100):
    """Benchmark a function over n iterations"""
    t0 = time.time()
    for _ in range(n):
        func()
    return (time.time() - t0) / n


# PyTorch timing
t_pytorch = time_it(lambda: model(x))

# ONNX Runtime timing
t_onnx = time_it(lambda: session.run(None, {session.get_inputs()[0].name: x.numpy()}))

print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(f"PyTorch avg time:      {t_pytorch*1000:.3f} ms")
print(f"ONNX Runtime avg time: {t_onnx*1000:.3f} ms")
print(f"Speedup:               {t_pytorch/t_onnx:.2f}x")