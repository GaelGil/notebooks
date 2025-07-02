import torch
from PIL import Image
import torch
from torchvision import transforms
from src.CNN import CNN  # or wherever your CNN class is defined
import src.config as config  # your config with parameters

transform = config.IMG_TRANSFORMATIONS
model = CNN(
    in_channels=config.IN_CHANNELS,
    num_classes=config.NUM_CLASSES,
    kernel_size=config.KERNEL_SIZE,
    dropout_rate=0.0  # or whatever you want for inference
).to(config.DEVICE)


model.load_state_dict(torch.load('./models/model_state_dict.pth', map_location=config.DEVICE))
model.eval()


dog_img_path = './dog_photo.png'
cat_img_path = './cat.png'
dog = Image.open(dog_img_path).convert('RGB')  
cat = Image.open(cat_img_path).convert('RGB')  

dog_tensor = transform(dog).unsqueeze(0).to(config.DEVICE)
cat_tensor = transform(cat).unsqueeze(0).to(config.DEVICE)
# input_tensor = input_tensor

with torch.no_grad():
    dog_output = model(dog_tensor)  # raw logits
    dog_probs = torch.sigmoid(dog_output)  # for binary classification (single output neuron)
    dog_prediction = (dog_probs > 0.5).long().item()

    cat_output = model(cat_tensor)  # raw logits
    cat_probs = torch.sigmoid(cat_output)  # for binary classification (single output neuron)
    cat_prediction = (cat_probs > 0.5).long().item()

print(f"Predicted class for dog photo: {dog_prediction} (probability: {dog_probs.item():.4f})")
print(f"Predicted class for cat photo: not {cat_prediction} (probability: {cat_probs.item():.4f})")


# Best Params: {'epoch': 10, 'lr': 0.001, 'dropout': 0.25}
# Test Accuracy: 0.7876849260295882