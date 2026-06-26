import config

from dataset import create_dataloaders

if __name__=="__main__":


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

    clips, labels = next(iter(train_loader))

    print("\nClip Shape :", clips.shape)
    print("Label Shape:", labels.shape)

    print("\nLabels:")
    print(labels)

    print("\nClip dtype:", clips.dtype)
    print("Label dtype:", labels.dtype)