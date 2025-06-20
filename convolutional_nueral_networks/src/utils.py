import torch
import torch.nn as nn
import logging
import os
import warnings
from PIL import Image

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
        epoch_losses = []
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            # if (i+1) % 10 == 0:  # print every 10 batches
            #     logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")



def evaluate(model, device, loader):
    """Function to evaluate our model.

    Args:
        model: The model we are trying to train.
        loader: The data
        device: The device we will use
    Returns:
        accuracy
    """
    model.eval()
    correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == y).sum().item()
            num_samples += y.size(0)

    accuracy = correct / num_samples
    return accuracy
