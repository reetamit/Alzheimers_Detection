from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import matplotlib.pyplot as plt
import base64


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your GitHub Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load("./efficientnet_b0.pth", map_location=device))
model = model.to(device)
model.eval()

# Define transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

gradients = []
activations = []

def save_gradient(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def save_activation(module, input, output):
    activations.append(output)

target_layer = model.features[6][1]
target_layer.register_forward_hook(save_activation)
target_layer.register_backward_hook(save_gradient)


# End points
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
    input_tensor = test_transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return {"label": int(predicted.item())}
    

@app.post("/explain")
async def explain(image: UploadFile = File(...)):
    gradients.clear()
    activations.clear()

    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
    input_tensor = test_transform(pil_image).unsqueeze(0).to(device)

    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Get gradients and activations
    grad = gradients[0].cpu().numpy()[0]
    act = activations[0].cpu().numpy()[0]

    # Compute Grad-CAM
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    # Resize CAM to original image size
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(pil_image.size, Image.BILINEAR)
    cam_img = np.array(cam_img)

    # Overlay CAM on original image
    original = np.array(pil_image)
    overlay = np.uint8(0.5 * original + 0.5 * cam_img)

    # Convert to base64
    fig, ax = plt.subplots()
    ax.imshow(overlay, cmap='gray')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "label": int(pred_class.item()),
        "gradcam": encoded
    }

