import torch
import torch.optim as optim
from torchvision.datasets import ImageFolder
from src import CNN, confing, train, accuracy

if __name__ == "__main__":
    # load in the dataset
    data_set = ImageFolder(confing.DATA_PATH, transform=confing.IMG_TRANSFORMATIONS)
    # get size of the entire dataset
    len_data_set = len(data_set)
    # get the size of train, validation, and test count
    train_count = int(0.7 * len_data_set)
    valid_count = int(0.2 * len_data_set)
    test_count = len_data_set - train_count - valid_count
    # slit the data into train, valid and test set
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(data_set, (train_count, valid_count, test_count))
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=confing.BATCH_SIZE, shuffle=True)
    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=confing.BATCH_SIZE, shuffle=True)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=confing.BATCH_SIZE, shuffle=True)

    # initialize the model
    model = CNN(in_channels=confing.IN_CHANNELS, clases=confing.NUM_CLASSES, kernel_size=confing.KERNEL_SIZE).to(confing.DEVICE)
    # set the optimizer
    optimizer = optim.Adam(model.parameters(), lr=confing.LEARNING_RATE)

    # train the model
    train(model=model, train_loader=train_dataset_loader, optimizer=optimizer, epochs=confing.EPOCHS, device=confing.DEVICE)

    # save model
    torch.save(confing.MODEL_PATH, model)