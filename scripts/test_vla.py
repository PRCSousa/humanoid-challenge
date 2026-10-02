import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

print(f"is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("No GPU")

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
n_params = sum(p.numel() for p in policy.parameters())
print(f"SmolVLA loaded: {n_params/1e6:.0f}M params")
print(f"Device: {next(policy.parameters()).device}")
