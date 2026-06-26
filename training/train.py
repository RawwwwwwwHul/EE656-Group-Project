import os
import random

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from tqdm import tqdm

import config

from architecture.istvt import ISTVT
from dataset import create_dataloaders

def set_seed(seed):
    """
    Sets all random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """
    Trains the model for one epoch.

    Returns
    -------
    epoch_loss : float

    epoch_accuracy : float
    """

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for clips, labels in progress_bar:

        clips = clips.to(device)

        labels = labels.to(device)

        # ---------------------------------------
        # Forward
        # ---------------------------------------

        optimizer.zero_grad(set_to_none=True)

        outputs = model(clips)

        loss = criterion(
            outputs,
            labels,
        )

        # ---------------------------------------
        # Backward
        # ---------------------------------------

        loss.backward()

        optimizer.step()

        # ---------------------------------------
        # Statistics
        # ---------------------------------------

        running_loss += loss.item()

        _, predictions = torch.max(
            outputs,
            dim=1,
        )

        total += labels.size(0)

        correct += (predictions == labels).sum().item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / total:.2f}%"
        )

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy

def validate(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Evaluates the model on the validation set.

    Returns
    -------
    epoch_loss : float

    epoch_accuracy : float
    """

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False,
    )

    with torch.no_grad():

        for clips, labels in progress_bar:

            clips = clips.to(device)

            labels = labels.to(device)

            # ---------------------------------------
            # Forward
            # ---------------------------------------

            outputs = model(clips)

            loss = criterion(
                outputs,
                labels,
            )

            # ---------------------------------------
            # Statistics
            # ---------------------------------------

            running_loss += loss.item()

            _, predictions = torch.max(
                outputs,
                dim=1,
            )

            total += labels.size(0)

            correct += (predictions == labels).sum().item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / total:.2f}%"
            )

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = 100 * correct / total

    return (epoch_loss, epoch_accuracy)

def main():

    # -------------------------------------------------
    # Reproducibility
    # -------------------------------------------------

    set_seed(config.SEED)

    device = config.DEVICE

    

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(

        root_dir=config.DATA_ROOT,

        batch_size=config.BATCH_SIZE,

        train_ratio=config.TRAIN_RATIO,

        val_ratio=config.VAL_RATIO,

        test_ratio=config.TEST_RATIO,

        seed=config.SEED,

        num_workers=config.NUM_WORKERS,

    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = ISTVT(

        num_blocks=config.NUM_BLOCKS,

        embed_dim=config.EMBED_DIM,

        num_heads=config.NUM_HEADS,

        mlp_ratio=config.MLP_RATIO,

        dropout=config.DROPOUT,

        num_classes=config.NUM_CLASSES,

    ).to(device)

    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable Parameters: {num_params:,}")

    # -------------------------------------------------
    # Loss Function
    # -------------------------------------------------

    num_fake = 6000
    num_real = 4500

    weights = torch.tensor(
        [
            num_fake / num_real,   # Weight for REAL (class 0)
            1.0                    # Weight for FAKE (class 1)
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    optimizer = AdamW(

        model.parameters(),

        lr=config.LEARNING_RATE,

        weight_decay=config.WEIGHT_DECAY,

    )

    # -------------------------------------------------
    # Create checkpoint directory
    # -------------------------------------------------

    os.makedirs(

        config.CHECKPOINT_DIR,

        exist_ok=True,

    )

    best_accuracy = 0.0

    # -------------------------------------------------
    # Training Loop
    # -------------------------------------------------

    for epoch in range(config.EPOCHS):

        print("=" * 70)

        print(

            f"Epoch {epoch + 1}/{config.EPOCHS}"

        )

        print("=" * 70)

        train_loss, train_acc = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device,

        )

        val_loss, val_acc = validate(

            model,

            val_loader,

            criterion,

            device,

        )

        print(

            f"Train Loss : {train_loss:.4f} | "

            f"Train Acc : {train_acc:.2f}%"

        )

        print(

            f"Val Loss   : {val_loss:.4f} | "

            f"Val Acc   : {val_acc:.2f}%"

        )

        # ---------------------------------------------
        # Save Best Model
        # ---------------------------------------------

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            torch.save(

                model.state_dict(),

                os.path.join(

                    config.CHECKPOINT_DIR,

                    config.BEST_MODEL_NAME,

                ),

            )

            print(

                f"Best model saved "

                f"({best_accuracy:.2f}%)"

            )

    # -------------------------------------------------
    # Save Final Model
    # -------------------------------------------------

    torch.save(

        model.state_dict(),

        os.path.join(

            config.CHECKPOINT_DIR,

            config.LAST_MODEL_NAME,

        ),

    )

    print("\nTraining Complete!")

    print(

        f"Best Validation Accuracy: "

        f"{best_accuracy:.2f}%"

    )

if __name__ == "__main__":

    main()