import torch

from architecture.istvt import ISTVT
from training import project_config as config
from training.dataset import create_dataloaders


device = config.DEVICE

(
    train_dataset,
    val_dataset,
    test_dataset,
    train_loader,
    val_loader,
    test_loader,
) = create_dataloaders(
    root_dir=config.DATA_ROOT,
    batch_size=1,
    train_ratio=config.TRAIN_RATIO,
    val_ratio=config.VAL_RATIO,
    test_ratio=config.TEST_RATIO,
    seed=config.SEED,
    num_workers=0,
)

model = ISTVT(
    num_blocks=config.NUM_BLOCKS,
    embed_dim=config.EMBED_DIM,
    num_heads=config.NUM_HEADS,
    mlp_ratio=config.MLP_RATIO,
    dropout=config.DROPOUT,
    num_classes=config.NUM_CLASSES,
).to(device)

state_dict = torch.load(
    "best_istvt.pth",
    map_location=device,
    weights_only=False,
)

model.load_state_dict(state_dict)

model.eval()

print("Model loaded.")

clips, labels = next(iter(test_loader))

clips = clips.to(device)
labels = labels.to(device)

print("Input:", clips.shape)


outputs = model(clips)

print("Output:", outputs)

score = outputs[0, outputs.argmax(dim=1)]

model.zero_grad()

score.backward()

print("\nChecking attention tensors...\n")

for i, block in enumerate(model.blocks):

    print("=" * 60)
    print(f"BLOCK {i}")

    temp_attn = block.temporal.attention.get_attention_map()
    temp_grad = block.temporal.attention.get_attention_gradients()

    spat_attn = block.spatial.attention.get_attention_map()
    spat_grad = block.spatial.attention.get_attention_gradients()

    print(
        "Temporal attention:",
        None if temp_attn is None else temp_attn.shape,
    )

    print(
        "Temporal gradients:",
        None if temp_grad is None else temp_grad.shape,
    )

    print(
        "Spatial attention:",
        None if spat_attn is None else spat_attn.shape,
    )

    print(
        "Spatial gradients:",
        None if spat_grad is None else spat_grad.shape,
    )

print("\nDone.")