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
        losses = []
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
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            # optimize
            optimizer.step() 
            # add loss to list of losses
            losses.append(loss.item())
    
        logger.info(f'Epoch: {epoch} ---- loss: {loss.item()}')
