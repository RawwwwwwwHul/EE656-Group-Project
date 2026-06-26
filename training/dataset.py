import os
import random
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
)


class ISTVTDataset(Dataset):
    """
    Dataset for the ISTVT model.

    Expected Directory Structure
    ----------------------------
    clips/
    │
    ├── fake/
    │     clip_0001.pt
    │     clip_0002.pt
    │     ...
    │
    └── real/
          clip_0001.pt
          clip_0002.pt
          ...

    Labels
    ------
    fake -> 1
    real -> 0
    """

    def __init__(
        self,
        file_paths,
    ):
        super().__init__()

        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        data = torch.load(self.file_paths[idx], map_location="cpu")

        clip = data["clip"].float().contiguous()

        # Expected shape:
        # (6,3,128,128)

        label = torch.tensor(data["label"], dtype=torch.long)

        return clip, label


def create_dataloaders(
    root_dir,
    batch_size=8,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    num_workers=2,
):
    """
    Creates train, validation and test DataLoaders.

    Parameters
    ----------
    root_dir : str
    Path to the clips directory.

    Returns
    -------
    train_dataset
    val_dataset
    test_dataset

    train_loader
    val_loader
    test_loader
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)

    fake_dir = os.path.join(root_dir, "fake")
    real_dir = os.path.join(root_dir, "real")

    fake_files = [
        os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(".pt")
    ]

    real_files = [
        os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(".pt")
    ]

    random.shuffle(fake_files)
    random.shuffle(real_files)

    # -------------------------------------------------------
    # Split Fake Clips
    # -------------------------------------------------------

    n_fake = len(fake_files)

    fake_train = int(train_ratio * n_fake)
    fake_val = int(val_ratio * n_fake)

    train_fake = fake_files[:fake_train]

    val_fake = fake_files[fake_train : fake_train + fake_val]

    test_fake = fake_files[fake_train + fake_val :]

    # -------------------------------------------------------
    # Split Real Clips
    # -------------------------------------------------------

    n_real = len(real_files)

    real_train = int(train_ratio * n_real)
    real_val = int(val_ratio * n_real)

    train_real = real_files[:real_train]

    val_real = real_files[real_train : real_train + real_val]

    test_real = real_files[real_train + real_val :]

    # -------------------------------------------------------
    # Combine
    # -------------------------------------------------------

    train_files = train_fake + train_real
    train_labels = [1] * len(train_fake) + [0] * len(train_real)

    val_files = val_fake + val_real
    val_labels = [1] * len(val_fake) + [0] * len(val_real)

    test_files = test_fake + test_real
    test_labels = [1] * len(test_fake) + [0] * len(test_real)

    # Shuffle each split

    train = list(zip(train_files, train_labels))
    val = list(zip(val_files, val_labels))
    test = list(zip(test_files, test_labels))

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    train_files, train_labels = zip(*train)
    val_files, val_labels = zip(*val)
    test_files, test_labels = zip(*test)

    # -------------------------------------------------------
    # Datasets
    # -------------------------------------------------------

    train_dataset = ISTVTDataset(train_files)

    val_dataset = ISTVTDataset(
        val_files,
    )

    test_dataset = ISTVTDataset(test_files)

    # -------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print(f"Training   : {len(train_dataset)} clips")
    print(f"Validation : {len(val_dataset)} clips")
    print(f"Testing    : {len(test_dataset)} clips")

    print()

    print(f"Fake clips : {len(fake_files)}")
    print(f"Real clips : {len(real_files)}")

    print("=" * 60)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )
