import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, models


# Load checkpoint
checkpoint = torch.load("best_model.pt", map_location="cpu")
idx_to_class = checkpoint["idx_to_class"]
num_classes = len(idx_to_class)

# Build model (ResNet50)
model = models.resnet50(weights=None)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Rebuild classifier EXACTLY as training
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)

# Load weights
model.load_state_dict(checkpoint["model_state"])
model.eval()


# Image transforms
IMG_SIZE = 160
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Prediction function
def predict(image):
    image = image.convert("RGB")
    x = val_transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    top_probs, top_idx = probs.topk(3, dim=1)

    top_probs = top_probs.numpy().squeeze()
    top_idx = top_idx.numpy().squeeze()

    return {
        idx_to_class[int(i)]: float(p)
        for i, p in zip(np.atleast_1d(top_idx), np.atleast_1d(top_probs))
    }


# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="🌿 Plant Disease Detection",
    description="Upload a plant leaf image to predict the disease using a ResNet-based deep learning model."
)

demo.launch(share = True)
