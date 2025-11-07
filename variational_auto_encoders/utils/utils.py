import torch
import torch.nn as nn
import logging
import os
import warnings
from PIL import Image
import src.config as config
from src.CNN import CNN

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
FEATURE_MAPS = {}


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
    criterion = nn.BCEWithLogitsLoss()  # set loss function for training
    model.train()  # set train mode

    for epoch in range(epochs):
        epoch_losses = []  # stores loss values for each batch in the current epoch

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # ensure labels are float and have shape [batch_size, 1]
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # outputs shape: [batch_size, 1], raw logits

            # loss bc even tho we are doing classification we are given
            # a error value/how wrong our model was
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(
            f"Epoch {epoch + 1} out of {epochs} completed. Average Loss: {avg_loss:.4f}"
        )


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
    correct = 0  # num correct samples
    num_samples = 0  # num samples

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


def load_model(model_path):
    """Load a model from a given path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = CNN(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        kernel_size=config.KERNEL_SIZE,
        dropout_rate=0.0,  # or whatever you want for inference
    ).to(config.DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    return model


def load_image(image_path):
    """Load an image from a given path and apply transformations."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = config.IMG_TRANSFORMATIONS(image).unsqueeze(0).to(config.DEVICE)
    return image_tensor


def predict(model, image_tensor):
    """Make a prediction on a single image tensor."""
    with torch.no_grad():
        output = model(image_tensor)  # raw logits
        probs = torch.sigmoid(
            output
        )  # for binary classification (single output neuron)
        prediction = (probs > 0.5).long().item()  # convert to 0 or 1

    return prediction, probs.item()  # return both prediction and probability


def test_sample(model, img_path):
    img_tensor = load_image(img_path)
    prediction, probability = predict(model, img_tensor)
    logger.info(
        f"Predicted class for {img_path}: {prediction} (probability: {probability:.4f})"
    )
    return prediction, probability
