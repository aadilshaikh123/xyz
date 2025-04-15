import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from scipy.stats import norm

# ---- USER CONFIGURATION ----
IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS", "UNKNOWN"]
MODEL_PATH = "best_model.pth"
TEST_DIR = "/kaggle/input/combined-unknown-pneumonia-and-tuberculosis/data/test"  # The main test directory (should have subfolders per class)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOTSTRAP = 1000  # For confidence intervals

# ---- IMAGE PREPROCESSING ----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---- LOAD MODEL ----
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(256, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ---- LOAD TEST DATA ----
def load_test_data(test_dir: str) -> Tuple[List[str], List[int]]:
    img_paths, labels = [], []
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir): continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_paths.append(os.path.join(class_dir, fname))
                labels.append(idx)
    return img_paths, labels

# ---- PREDICT ----
def predict(model, img_paths: List[str]) -> List[int]:
    preds = []
    with torch.no_grad():
        for path in img_paths:
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            output = model(img_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            preds.append(pred_idx)
    return preds

# ---- BOOTSTRAP CONFIDENCE INTERVALS ----
def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, alpha=0.05):
    rng = np.random.default_rng()
    stats = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        stats.append(metric_func(y_true[idx], y_pred[idx]))
    stats = np.sort(stats)
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper

# ---- METRIC FUNCTIONS ----
def print_metrics(y_true, y_pred):
    print("\n--- CLASSIFICATION REPORT ---")
    # Ensure all metrics arrays have length == len(CLASS_NAMES)
    prec = precision_score(y_true, y_pred, labels=range(len(CLASS_NAMES)), average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=range(len(CLASS_NAMES)), average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=range(len(CLASS_NAMES)), average=None, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"\nClass: {name}")
        print(f"  Precision: {prec[i]:.4f}")
        print(f"  Recall:    {rec[i]:.4f}")
        print(f"  F1-score:  {f1[i]:.4f}")

    # Overall metrics
    print(f"\nMacro Precision: {precision_score(y_true, y_pred, labels=range(len(CLASS_NAMES)), average='macro', zero_division=0):.4f}")
    print(f"Macro Recall:    {recall_score(y_true, y_pred, labels=range(len(CLASS_NAMES)), average='macro', zero_division=0):.4f}")
    print(f"Macro F1-score:  {f1_score(y_true, y_pred, labels=range(len(CLASS_NAMES)), average='macro', zero_division=0):.4f}")

    # Confidence Intervals
    print("\n--- BOOTSTRAPPED 95% CONFIDENCE INTERVALS ---")
    acc_ci = bootstrap_metric(y_true, y_pred, accuracy_score, N_BOOTSTRAP)
    print(f"Accuracy 95% CI: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
    for i, name in enumerate(CLASS_NAMES):
        # For per-class precision and recall, define per-class metric as "binary accuracy" for that class
        def prec_func(t, p): return precision_score(t, p, labels=[1], average=None, zero_division=0)[0] if np.any((np.array(t)==1)+(np.array(p)==1)) else 0.0
        def rec_func(t, p): return recall_score(t, p, labels=[1], average=None, zero_division=0)[0] if np.any(np.array(t)==1) else 0.0
        y_true_bin = (np.array(y_true) == i).astype(int)
        y_pred_bin = (np.array(y_pred) == i).astype(int)
        prec_ci = bootstrap_metric(y_true_bin, y_pred_bin, prec_func, N_BOOTSTRAP)
        rec_ci = bootstrap_metric(y_true_bin, y_pred_bin, rec_func, N_BOOTSTRAP)
        print(f"{name}: Precision 95% CI: [{prec_ci[0]:.4f}, {prec_ci[1]:.4f}]")
        print(f"{name}: Recall 95% CI:    [{rec_ci[0]:.4f}, {rec_ci[1]:.4f}]")

# ---- PLOTTING ----
def plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def plot_metric_bars(y_true, y_pred):
    prec = precision_score(y_true, y_pred, labels=range(len(CLASS_NAMES)), average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=range(len(CLASS_NAMES)), average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=range(len(CLASS_NAMES)), average=None, zero_division=0)
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    plt.figure(figsize=(8, 5))
    plt.bar(x - width, prec, width, label="Precision")
    plt.bar(x, rec, width, label="Recall")
    plt.bar(x + width, f1s, width, label="F1-score")
    plt.xticks(x, CLASS_NAMES)
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Per-Class Metrics")
    plt.tight_layout()
    plt.show()

# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    print("Loading model...")
    model = load_model()
    print("Loading test data...")
    img_paths, y_true = load_test_data(TEST_DIR)

    print(f"Found {len(img_paths)} test images.")

    if len(img_paths) == 0:
        print("No test images found. Please check your TEST_DIR and class folders.")
        exit(1)

    print("Predicting on test set...")
    y_pred = predict(model, img_paths)

    print_metrics(y_true, y_pred)
    plot_confusion(y_true, y_pred, CLASS_NAMES)
    plot_metric_bars(y_true, y_pred)

    # Show class distribution in the test set
    print("\nClass distribution in the test set:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name}: {np.sum(np.array(y_true)==i)} samples")