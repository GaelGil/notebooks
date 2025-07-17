import torch.nn as nn


def initialize_weights(model):
    """Function to initialize weights

    Args:
        model: The model whose weights we want to initialize

    Returns:
        None
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
