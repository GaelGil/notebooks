import torch
import torch.nn as nn
import logging

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train(model, train_loader, optimizer, epochs, device):
    """Function to train our convolutinal neural network.

    Args:
        model: The model we are trying to train.
        train_loader: The data for training.
        epochs: How many iterations of training we are going to do.
        optimizer: The optimizer for our network
    Returns:
        None
    """
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        accuracies = []
        for i, (inputs, labels) in enumerate(train_loader):
            # get inputs and labels
            inputs, labels = inputs.to(device), labels.to(device)
            # turn the labels of the batch from [0, 1] to [[0], [1]]
            # also makes them floats
            labels = labels.unsqueeze(1).float()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward 
            outputs = model(inputs)
            # calculate loss
            accuracy = criterion(outputs, labels)
            # backward
            accuracy.backward()
            # optimize
            optimizer.step() 
            # add loss to list of losses
            accuracies.append(accuracy.item())
    
        logger.info(f'Epoch: {epoch} ---- loss: {accuracy.item()}')



def evaluate(loader, model, device):
    """Function to evaluate our model.

    Args:
        model: The model we are trying to train.
        loader: The data
        device: The device we will use
    Returns:
        accuracy
    """
    correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            output = model(x)
            _, predicted = torch.max(output, 1)
            correct += (predicted == y).sum().item()
            num_samples += y.size(0)

    return correct / num_samples
