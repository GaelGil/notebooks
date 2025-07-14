import torch
from PIL import Image
from src.CNN import CNN
import src.config as config
import matplotlib.pyplot as plt
import math

transform = config.IMG_TRANSFORMATIONS
model = CNN(
    in_channels=config.IN_CHANNELS,
    num_classes=config.NUM_CLASSES,
    kernel_size=config.KERNEL_SIZE,
    dropout_rate=0.25,
).to(config.DEVICE)

model.load_state_dict(torch.load("./models/model_state_dict.pth"))
model.eval()

# dictionary to hold feature maps
feature_maps = {}


# hook to capture feature maps
def get_activation(name):
    def hook(model, input, output):
        feature_maps[name] = output

    return hook


# register hooks for each layer
model.conv1.register_forward_hook(get_activation("conv1"))
model.pool1.register_forward_hook(get_activation("pool1"))
model.conv2.register_forward_hook(get_activation("conv2"))
model.pool2.register_forward_hook(get_activation("pool2"))
model.conv3.register_forward_hook(get_activation("conv3"))
model.pool3.register_forward_hook(get_activation("pool3"))


# function to visualize all feature maps in a grid format
def show_all_feature_maps(tensor, title, max_cols=8):
    tensor = tensor.detach().cpu()
    channels = tensor.shape[1]
    cols = min(max_cols, channels)
    rows = math.ceil(channels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(title)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        if i < channels:
            ax.imshow(tensor[0, i], cmap="gray")
            ax.set_title(f"Channel {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# Load an image and pass it through the model to get feature maps
cat_img_path = "./samples/cat/cat_photo.png"
cat = Image.open(cat_img_path).convert("RGB")
cat_tensor = transform(cat).unsqueeze(0).to(config.DEVICE)
_ = model(cat_tensor)  # forward pass to get feature maps
show_all_feature_maps(feature_maps["pool1"], title="Pool1 Feature Maps")

dog_img_path = "./samples/dog/dog_photo.png"
dog = Image.open(dog_img_path).convert("RGB")
dog_tensor = transform(dog).unsqueeze(0).to(config.DEVICE)
dog_output = model(dog_tensor)  # raw logits
show_all_feature_maps(feature_maps["pool3"], title="Pool1 Feature Maps")
