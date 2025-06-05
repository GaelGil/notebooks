import torch 

def check_accuracy(loader, model, device):
    """Function to train our convolutinal neural network.

    Args:
        model: The model we are trying to train.
        loader: The data
        device: The device we will use
    Returns:
        None
    """
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples