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
    criterion = nn.BCEWithLogitsLoss() # set loss function for training
    model.train()  # set train mode
    
    for epoch in range(epochs):
        # epoch_losses = [] # stores loss values for each batch in the current epoch
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # ensure labels are float and have shape [batch_size, 1]
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            labels = labels.float()
            
            optimizer.zero_grad()
            outputs = model(inputs)  # outputs shape: [batch_size, 1], raw logits
            
            # loss bc even tho we are doing classification we are given 
            # a error value/how wrong our model was
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            
            # epoch_losses.append(loss.item())
        
            # avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f'Epoch {epoch+1} out of {epochs} completed, Loss at this epoch was: {loss}')



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
    correct = 0 # num correct samples
    num_samples = 0 # num samples 

    with torch.no_grad():
        for x, y in loader: 
            x = x.to(device)
            y = y.to(device)

            # shape: [batch_size, 1] ie batch size 4: torch.tensor([[0.2],[-1.5],[0.7],[-0.3]])
            outputs = model(x)  
             # apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs) 
            # if prob is greater 0.5 true else false, .long() turns them inot 0 and 1
            # squeeze changess ([[0.2],[-1.5],[0.7],[-0.3]]) to [0.2, -1.5, 0.7, -0.3]
            predicted = (probs > 0.5).long().squeeze(1) 

            # compare predicted with truth, sum the correct number of truth (all ones)
            # .item makes it a single int instead of torch.tensor([int])
            correct += (predicted == y).sum().item()
            num_samples += y.size(0)

    accuracy = correct / num_samples
    return accuracy
