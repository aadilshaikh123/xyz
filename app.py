import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

# --- Settings ---
IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS", "UNKNOWN"]
MODEL_PATH = "best_model.pth"

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Grad-CAM Utility Functions ---
def generate_gradcam(model, img_tensor, class_idx):
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output.cpu().data.numpy())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].cpu().data.numpy())

    handle_fw = model.layer4.register_forward_hook(forward_hook)
    handle_bw = model.layer4.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(img_tensor)
    class_score = output[0, class_idx]
    class_score.backward()

    grads = gradients[0][0]
    fmap = features[0][0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img_tensor.shape[2], img_tensor.shape[3]))
    if np.max(cam) != 0:
        cam -= np.min(cam)
        cam /= np.max(cam)
    handle_fw.remove()
    handle_bw.remove()
    return cam

def overlay_cam_on_image(img: Image.Image, cam: np.ndarray, alpha=0.4) -> Image.Image:
    img = np.array(img.resize(cam.shape[::-1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlayed = heatmap * alpha + img * (1 - alpha)
    overlayed = np.uint8(overlayed)
    return Image.fromarray(overlayed)

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(256, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --- Streamlit UI ---
st.title("Chest X-Ray Multi-Class Classifier")
st.write(
    "Upload, drag & drop, paste *into file dialog*, or take a photo of a chest X-ray image. "
    "The model will classify it as NORMAL, PNEUMONIA, TUBERCULOSIS, or UNKNOWN."
)

tab1, tab2 = st.tabs(["Upload / Paste", "Camera"])

image = None

# --- Upload or paste image ---
with tab1:
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (drag/drop, browse, or paste into the dialog)", 
        type=["jpg", "jpeg", "png", "bmp"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

# --- Camera input ---
with tab2:
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# --- Prediction ---
if image is not None:
    st.image(image, caption='Input Image', use_column_width=True)
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze().numpy()
        pred_idx = np.argmax(probabilities)
        pred_class = CLASS_NAMES[pred_idx]
    st.markdown(f"### Prediction: **{pred_class}**")
    st.markdown("#### Probability Scores:")
    for idx, cname in enumerate(CLASS_NAMES):
        st.write(f"{cname}: {probabilities[idx]:.4f}")
    st.bar_chart({c: [probabilities[i]] for i, c in enumerate(CLASS_NAMES)})

    # --- Grad-CAM Button and Display ---
    if st.button("Show Grad-CAM Heatmap"):
        with st.spinner("Generating Grad-CAM..."):
            cam = generate_gradcam(model, img_tensor, pred_idx)
            cam_image = overlay_cam_on_image(image, cam)
        st.image(cam_image, caption=f"Grad-CAM for {pred_class}", use_column_width=True)

# --- Extra Info ---
with st.expander("ℹ️ Model & Class Information"):
    st.write("""
    **Model:** ResNet50, trained for multi-class classification on chest X-ray images.

    - **NORMAL:** No signs of pneumonia or tuberculosis.
    - **PNEUMONIA:** Signs of pneumonia detected.
    - **TUBERCULOSIS:** Signs of tuberculosis detected.
    - **UNKNOWN:** Image does not fit any of the above categories or is unclear.

    **How to use:**  
    - Upload an image, drag and drop, or take a photo.
    - You can paste a copied image into the file dialog in most browsers.
    - After prediction, click "Show Grad-CAM Heatmap" to visualize model attention.
    """)