# python/tests/test_ptq.py
import os, json, math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# -----------------------------
# 0) Paths / Config
# ----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root
CKPT_PATH = os.path.join(ROOT, "data", "checkpoints", "mnist_fp32.pt")
CALIB_BATCH = 64           # images used to compute activation scales (per-tensor)
EVAL_SAMPLES = 10          # images to compare FP32 vs INT8-emulated top-1
SEED = 42
DEVICE = "cpu"             # keep deterministic/portable

torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------------
# 1) Model (must match training)
# ----------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1   = nn.Linear(64 * 12 * 12, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)       # logits
        return x

# ----------------------------
# 2) Load FP32 checkpoint
# ----------------------------
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

fp32_model = Net().to(DEVICE).eval()
fp32_model.load_state_dict(state_dict)

# ----------------------------
# 3) Data (MNIST test set + same normalization as training)
# ----------------------------
mean, std = 0.1307, 0.3081
tfm = transforms.Compose([transforms.ToTensor()])   # normalize manually to control dtype
test_set = datasets.MNIST(root=os.path.join(ROOT, "data"), train=False, download=True, transform=tmf if (tmf:=tfm) else tfm)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=CALIB_BATCH, shuffle=False, num_workers=0)

# Pull one calibration batch (also contains our eval samples at the front)
calib_imgs, calib_labels = next(iter(test_loader))  # (N,1,28,28)
x_calib = ((calib_imgs.float() - mean) / std).to(DEVICE)

# ----------------------------
# 4) Helpers: symmetric INT8 quantization (per tensor)
# -------em---------------------
def get_symm_scale(t: torch.Tensor) -> float:
    maxabs = t.abs().max().item()
    return max(maxabs / 127.0, 1e-12)

def quantize_tensor_symm(t: torch.Tensor, scale: float) -> torch.Tensor:
    # emulate int8 via clamp/round then dequant (fake-quant for activations)
    q = torch.clamp(torch.round(t / scale), -128, 127)
    return q * scale

def quantize_weights_symm_per_layer(w: torch.Tensor) -> (torch.Tensor, float):
    scale = get_symm_scale(w)
    q = torch.clamp(torch.round(w / scale), -128, 127)
    w_q = q * scale
    return w_q, scale

# ----------------------------
# 5) Build an INT8-emulated model
#    - Per-layer symmetric weight quantization
#    - Per-tensor symmetric activation quantization at each layer
# ----------------------------
class NetINT8Emu(nn.Module):
    def __init__(self, fp32: Net):
        super().__init__()
        # Copy layer structures
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=True)
        self.fc1   = nn.Linear(64 * 12 * 12, 128, bias=True)
        self.fc2   = nn.Linear(128, 10, bias=True)

        # Copy FP32 params first (for bias values etc.)
        self.load_state_dict(fp32.state_dict(), strict=True)

        # Quantize weights per-layer (symmetric)
        with torch.no_grad():
            self.conv1.weight.data, self.Sw_conv1 = quantize_weights_symm_per_layer(self.conv1.weight.data)
            self.conv2.weight.data, self.Sw_conv2 = quantize_weights_symm_per_layer(self.conv2.weight.data)
            self.fc1.weight.data,   self.Sw_fc1   = quantize_weights_symm_per_layer(self.fc1.weight.data)
            self.fc2.weight.data,   self.Sw_fc2   = quantize_weights_symm_per_layer(self.fc2.weight.data)
        # Note: biases left as FP32 (standard practice in many INT8 deployments)

        # Activation scales (per-tensor) — compute from calibration batch below
        self.Sx_in    = None
        self.Sx_c1out = None
        self.Sx_c2out = None
        self.Sx_fc1in = None
        self.Sx_fc2in = None

    def calibrate(self, x: torch.Tensor):
        # Compute per-tensor scales at each activation point using calibration data
        with torch.no_grad():
            # Input scale
            self.Sx_in = get_symm_scale(x)

            # After conv1 + ReLU
            y1 = F.relu(self.conv1(x))
            self.Sx_c1out = get_symm_scale(y1)

            # After conv2 + ReLU
            y2 = F.relu(self.conv2(quantize_tensor_symm(y1, self.Sx_c1out)))
            self.Sx_c2out = get_symm_scale(y2)

            # After pool + flatten
            y2p = F.max_pool2d(quantize_tensor_symm(y2, self.Sx_c2out), 2)
            y2f = torch.flatten(y2p, 1)
            self.Sx_fc1in = get_symm_scale(y2f)

            # After fc1 + ReLU
            y3 = F.relu(self.fc1(quantize_tensor_symm(y2f, self.Sx_fc1in)))
            self.Sx_fc2in = get_symm_scale(y3)

    def forward(self, x):
        # Fake-quantize activations per tensor at each step (symmetric INT8 emu)
        x = quantize_tensor_symm(x, self.Sx_in)
        x = F.relu(self.conv1(x))
        x = quantize_tensor_symm(x, self.Sx_c1out)

        x = F.relu(self.conv2(x))
        x = quantize_tensor_symm(x, self.Sx_c2out)

        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = quantize_tensor_symm(x, self.Sx_fc1in)

        x = F.relu(self.fc1(x))
        x = quantize_tensor_symm(x, self.Sx_fc2in)

        x = self.fc2(x)  # logits (typically left un-quantized or quantized to int32 accum then scaled)
        return x

# Instantiate emu model and calibrate
int8emu = NetINT8Emu(fp32_model).to(DEVICE).eval()
int8emu.calibrate(x_calib)   # computes Sx_* using calibration batch

# ----------------------------
# 6) Evaluate: FP32 vs INT8 emu on first EVAL_SAMPLES
# ----------------------------
# Prepare eval slice (first EVAL_SAMPLES images from the calibration batch for determinism)
x_eval = x_calib[:EVAL_SAMPLES]
y_eval = calib_labels[:EVAL_SAMPLES]

with torch.no_grad():
    logits_fp32 = fp32_model(x_eval)
    logits_int8 = int8emu(x_eval)

pred_fp32 = logits_fp32.argmax(dim=1)
pred_int8 = logits_int8.argmax(dim=1)

match_mask = (pred_fp32 == pred_int8)
num_match = int(match_mask.sum().item())

print("=== INT8 Emulation vs FP32 (Top-1) ===")
print(f"Samples compared: {EVAL_SAMPLES}")
print(f"Top-1 agreement : {num_match}/{EVAL_SAMPLES}")
print(f"FP32 preds      : {pred_fp32.cpu().numpy().tolist()}")
print(f"INT8 emu preds  : {pred_int8.cpu().numpy().tolist()}")
print(f"Ground truth    : {y_eval.cpu().numpy().tolist()}")

# Optional: small log of scale values for traceability
scales_log = {
    "Sx_in": float(int8emu.Sx_in),
    "Sx_c1out": float(int8emu.Sx_c1out),
    "Sx_c2out": float(int8emu.Sx_c2out),
    "Sx_fc1in": float(int8emu.Sx_fc1in),
    "Sx_fc2in": float(int8emu.Sx_fc2in),
    "Sw_conv1": float(int8emu.Sw_conv1),
    "Sw_conv2": float(int8emu.Sw_conv2),
    "Sw_fc1": float(int8emu.Sw_fc1),
    "Sw_fc2": float(int8emu.Sw_fc2),
}
print("\nActivation/Weight scales (summary):")
for k, v in scales_log.items():
    print(f"  {k:>8s} = {v:.6g}")

# ----------------------------
# 7) Assertions for DoD
# ----------------------------
# DoD requirement: "FP32 vs INT8 emu top-1 unchanged on 10 samples"
assert num_match == EVAL_SAMPLES, (
    f"Top-1 mismatch: {num_match}/{EVAL_SAMPLES} matched. "
    f"Investigate scales/quant ranges or increase CALIB_BATCH."
)

print("\n[PASS] PTQ emulation test: FP32 and INT8-emulated top-1 are identical on the first "
      f"{EVAL_SAMPLES} samples.")
