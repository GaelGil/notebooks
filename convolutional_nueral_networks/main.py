import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from src import config
from src.CNN import CNN
from src.utils import train, evaluate

if __name__ == "__main__":
    # load in the dataset
    data_set = ImageFolder(config.DATA_PATH, transform=config.IMG_TRANSFORMATIONS)
    # get size of the entire dataset
    len_data_set = len(data_set)
    # get the size of train, validation, and test count
    train_count = int(0.7 * len_data_set)
    valid_count = int(0.2 * len_data_set)
    test_count = len_data_set - train_count - valid_count
    # slit the data into train, valid and test set
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(data_set, (train_count, valid_count, test_count))
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, pin_memory=True)
    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, pin_memory=True)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True, pin_memory=True)

    # train and evaluate the model and select the best params
    best_accuracy = 0
    best_params = {}
    for lr in config.LR_RATES:
        for dropout in config.DROPOUT_RATES:
            model = CNN(in_channels=config.IN_CHANNELS,
            num_classes=config.NUM_CLASSES,
            kernel_size=config.KERNEL_SIZE,
            dropout_rate=dropout).to(config.DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            train(model=model, train_loader=train_dataset_loader, optimizer=optimizer, epochs=config.EPOCHS, device=config.DEVICE)
            eval_accuracy = evaluate(model=model, device=config.DEVICE, loader=valid_dataset_loader)
            # print(f'Epochs: {epoch}, LR {lr}, Dropout: {dropout}, Val Accuracy: {eval_accuracy:.4f}')
            if eval_accuracy > best_accuracy:
                best_acc = eval_accuracy
                best_params = {'epoch': config.EPOCHS, 'lr': lr, 'dropout': dropout}


    # initialize the model
    model = CNN(in_channels=config.IN_CHANNELS,
                num_classes=config.NUM_CLASSES,
                kernel_size=config.KERNEL_SIZE,
                dropout_rate=best_params['dropout']).to(config.DEVICE)
    
    # set the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.best_params['lr'])

    # train the model
    train(model=model,
          train_loader=train_dataset_loader,
          optimizer=optimizer,
          epochs=best_params['epoch'],
          device=config.DEVICE)


    # evalutate model 
    evaluate(loader=test_dataset_loader, model=model, device=config.DEVICE)

    # save model
    torch.save(model.state_dict(), config.MODEL_PATH)

