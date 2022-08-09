

# image transformations
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor(),
   transforms.Normalize((0.5), (0.5))
    ])

# loading the whole mnist dataset
DATA = torchvision.datasets.MNIST(
    root="./data/",
    train=False,
    download=False,
    transform=TRANSFORM_IMG
)
