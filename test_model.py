import torch

from architecture.istvt import ISTVT
from training import project_config

# -------------------------------
# Device
# -------------------------------

device = project_config.DEVICE

# -------------------------------
# Build model
# -------------------------------

model = ISTVT(
    num_blocks=project_config.NUM_BLOCKS,
    embed_dim=project_config.EMBED_DIM,
    num_heads=project_config.NUM_HEADS,
    mlp_ratio=project_config.MLP_RATIO,
    dropout=project_config.DROPOUT,
    num_classes=project_config.NUM_CLASSES,
).to(device)

# -------------------------------
# Load checkpoint
# -------------------------------

checkpoint_path = "best_istvt.pth"

state_dict = torch.load(
    checkpoint_path,
    map_location=device,
)

model.load_state_dict(state_dict)

model.eval()

print("=" * 60)
print("Model loaded successfully!")
print(f"Checkpoint : {checkpoint_path}")
print(f"Device     : {device}")
print("=" * 60)