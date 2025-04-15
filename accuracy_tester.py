import torch
from PIL import Image
from torchvision import transforms

# --- Settings ---
IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS", "UNKNOWN"]
MODEL_PATH = "best_model.pth"  # or "final_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Image Preprocessing (must match training/val transforms, except for augmentation) ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Function to Predict a Single Image ---
def predict_image(image_path, model, transform, class_names, device):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    return predicted_class, probabilities

# --- Load Model ---
from torchvision import models
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(256, len(CLASS_NAMES))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)

# --- Usage Example ---
image_path = "/kaggle/input/combined-unknown-pneumonia-and-tuberculosis/data/test/TUBERCULOSIS/76.png"  # <-- change this to your image path
pred_class, probs = predict_image(image_path, model, transform, CLASS_NAMES, DEVICE)
print(f"Predicted class: {pred_class}")
print("Class probabilities:")
for i, cname in enumerate(CLASS_NAMES):
    print(f"  {cname}: {probs[i]:.4f}")