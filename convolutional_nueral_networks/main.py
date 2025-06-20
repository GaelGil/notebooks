import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from src import confing
from src.CNN import CNN
from src.utils import train, evaluate

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

    # train and evaluate the model and select the best params
    best_accuracy = 0
    best_params = {}
    for epoch in confing.EPOCHS:
        for lr in confing.LR_RATES:
            for dropout in confing.DROPOUT_RATES:
                model = CNN(in_channels=confing.IN_CHANNELS,
                num_classes=confing.NUM_CLASSES,
                kernel_size=confing.KERNEL_SIZE,
                dropout_rate=dropout).to(confing.DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()
                train(model=model, train_loader=train_dataset_loader, optimizer=optimizer, epochs=epoch, device=confing.DEVICE)
                eval_accuracy = evaluate(model=model, device=confing.DEVICE, val_loader=valid_dataset_loader)
                print(f'Epochs: {epoch}, LR {lr}, Dropout: {dropout}, Val Accuracy: {eval_accuracy:.4f}')
                if eval_accuracy > best_accuracy:
                    best_acc = eval_accuracy
                    best_params = {'epoch': epoch, 'lr': lr, 'dropout': dropout}


    # initialize the model
    model = CNN(in_channels=confing.IN_CHANNELS,
                num_classes=confing.NUM_CLASSES,
                kernel_size=confing.KERNEL_SIZE,
                dropout_rate=best_params['dropout']).to(confing.DEVICE)
    
    # set the optimizer
    optimizer = optim.Adam(model.parameters(), lr=confing.best_params['lr'])

    # train the model
    train(model=model,
          train_loader=train_dataset_loader,
          optimizer=optimizer,
          epochs=best_params['epoch'],
          device=confing.DEVICE)


    # evalutate model 
    evaluate(loader=test_dataset_loader, model=model, device=confing.DEVICE)

    # save model
    torch.save(model.state_dict(), confing.MODEL_PATH)

