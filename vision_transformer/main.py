from utils.config import config
from utils.ImageDataset import ImageDataset
from utils.train import train


def main():
    dataset = ImageDataset(dataset_path=config.DATA_PATH, transformations=None)
    dataset.split_data(
        train_split=config.TRAIN_SPLIT,
        val_split=config.VAL_SPLIT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    train()


if __name__ == "__main__":
    main()
