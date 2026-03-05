"""
Environment & GPU Checker
=========================
Run this FIRST to verify your installation before training.

  python check_env.py
"""
import sys
import importlib

print("=" * 60)
print("  Pediatric Cardiac Screening — Environment Check")
print("=" * 60)

# Python
print(f"\nPython : {sys.version.split()[0]}")

# Core packages
packages = [
    ("torch",        "PyTorch"),
    ("torchvision",  "TorchVision"),
    ("librosa",      "Librosa"),
    ("cv2",          "OpenCV"),
    ("numpy",        "NumPy"),
    ("pandas",       "Pandas"),
    ("scipy",        "SciPy"),
    ("fastapi",      "FastAPI"),
    ("streamlit",    "Streamlit"),
    ("PIL",          "Pillow"),
    ("tensorboard",  "TensorBoard"),
]

all_ok = True
for mod, name in packages:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  ✓ {name:<15} {ver}")
    except ImportError:
        print(f"  ✗ {name:<15} NOT INSTALLED")
        all_ok = False

# GPU
print()
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU            {gpu}  ({vram:.1f} GB VRAM)")
        print(f"  ✓ CUDA           {torch.version.cuda}")
        print(f"  ✓ cuDNN          {torch.backends.cudnn.version()}")
    else:
        print("  ⚠ GPU           Not available — training will use CPU (slow)")
except Exception:
    pass

# Project imports
print()
try:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from models.crnn_heart_sound import CRNN2D
    from models.nts_net_ultrasound import NTSNet
    from models.efficientnet_xray import EfficientNetV2XRay
    from models.gmu_fusion import MultimodalModel
    print("  ✓ Project imports OK")
except Exception as e:
    print(f"  ✗ Project import failed: {e}")
    all_ok = False

print()
if all_ok:
    print("✅ Environment is ready. Proceed to data preparation.")
else:
    print("❌ Fix the issues above before training.")
    print("   Run:  pip install -r requirements.txt")
print("=" * 60)
