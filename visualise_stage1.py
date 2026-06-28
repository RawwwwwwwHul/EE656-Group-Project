import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from architecture.istvt import ISTVT
from training.dataset import create_dataloaders
from training import project_config as config

device = config.DEVICE


model = ISTVT(
    num_blocks=config.NUM_BLOCKS,
    embed_dim=config.EMBED_DIM,
    num_heads=config.NUM_HEADS,
    mlp_ratio=config.MLP_RATIO,
    dropout=config.DROPOUT,
    num_classes=config.NUM_CLASSES,
).to(device)

state = torch.load(
    "best_istvt.pth",
    map_location=device,
)

model.load_state_dict(state)

model.eval()

print("Model loaded.")

(
    _,
    _,
    test_dataset,
    _,
    _,
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

for idx, (clip, label) in enumerate(test_loader):

    if idx == 10:
        break

    os.makedirs("outputs", exist_ok=True)

    sample_dir = os.path.join("outputs", f"sample_{idx}")
    os.makedirs(sample_dir, exist_ok=True)

    clip = clip.to(device)
    label = label.to(device)

    print("Loaded clip:", clip.shape)

    output = model(clip)

    prediction = output.argmax(dim=1)

    print("Prediction:", prediction.item())

    score = output[0, prediction]

    model.zero_grad()

    score.backward()

    print("Backward complete.")

    rollout = torch.eye(65, device=device)

    print("\nComputing rollout...")

    for block_idx, block in enumerate(model.blocks):

        attn = block.spatial.attention.get_attention_map()
        grad = block.spatial.attention.get_attention_gradients()

        if attn is None:
            raise RuntimeError(f"Block {idx}: attention map missing")

        if grad is None:
            raise RuntimeError(f"Block {idx}: attention gradients missing")

        print(f"Block {block_idx}")
        print("attn :", attn.shape)
        print("grad :", grad.shape)

        cam = attn * grad

        cam = cam.mean(dim=1)

        cam = torch.relu(cam)

        cam = cam.mean(dim=0)


        cam = cam + torch.eye(65, device=device)

        cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-6)

        rollout = cam @ rollout

    print("\nRollout complete.")

    mask = rollout[0, 1:]

    mask = mask.reshape(8, 8)

    mask = mask.detach().cpu().numpy()

    mask -= mask.min()

    mask /= (mask.max() + 1e-8)

    print("Mask shape:", mask.shape)

    os.makedirs("outputs", exist_ok=True)

    frames = clip[0].detach().cpu().permute(0, 2, 3, 1).numpy()

    heatmap = cv2.resize(
        mask,
        (128, 128),
        interpolation=cv2.INTER_CUBIC,
    )

    heatmap = np.clip(heatmap, 0, 1)

    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )


    cv2.imwrite(
        os.path.join(sample_dir, f"spatial_heatmap.png"),
        heatmap_color,
    )

    for i in range(6):

        frame = frames[i]

        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

        overlay = cv2.addWeighted(
            frame,
            0.60,
            heatmap_color,
            0.40,
            0,
        )

        cv2.imwrite(
            os.path.join(sample_dir, f"frame_{i}.png"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        )

        cv2.imwrite(
            os.path.join(sample_dir, f"overlay_{i}.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

    print("Saved frame overlays.")

    temporal_scores = []

    for block in model.blocks:

        attn = block.temporal.attention.get_attention_map()
        grad = block.temporal.attention.get_attention_gradients()

        cam = torch.relu(attn * grad)

        cam = cam.mean(dim=1)

        # (65,6,6)

        cam = cam.mean(dim=0)

        temporal_scores.append(
            cam.diag().detach().cpu().numpy()
        )

    temporal_scores = np.stack(
        temporal_scores,
        axis=0,
    )

    plt.figure(figsize=(8,4))

    plt.imshow(
        temporal_scores,
        aspect="auto",
    )

    plt.xlabel("Frame")

    plt.ylabel("Transformer Block")

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        os.path.join(sample_dir, f"temporal_scores.png"),
        dpi=300,
    )

    plt.close()

print("\nEverything saved to outputs/")