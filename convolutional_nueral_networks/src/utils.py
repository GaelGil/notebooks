import torch
import torch.nn as nn
import logging

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train(model, train_loader, optimizer, epochs, device):
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

            if (i+1) % 10 == 0:  # print every 10 batches
                print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")




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
