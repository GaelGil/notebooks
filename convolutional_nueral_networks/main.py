import torch
import torch.optim as optim
from torchvision.datasets import ImageFolder
from src import CNN, confing, train


if __name__ == "__main__":
    data_set = ImageFolder(confing.DATA_PATH, transform=confing.IMG_TRANSFORMATIONS)
    # get size of the entire dataset
    len_data_set = len(data_set)
    # get the size of train, validation, and test count
    train_count = int(0.7 * len_data_set)
    valid_count = int(0.2 * len_data_set)
    test_count = len_data_set - train_count - valid_count
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(data_set, (train_count, valid_count, test_count))
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

    # initialize the model
    model = CNN().to(confing.DEVICE)
    # set the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train the model
    train(model=model, train_loader=train_dataset_loader, optimizer=optimizer, epochs=2)