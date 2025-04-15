# Chest X-ray Classification with Statistical Inference

## Overview

This project is a deep learning-based web application for classifying chest X-ray images into four classes: **NORMAL**, **PNEUMONIA**, **TUBERCULOSIS**, and **UNKNOWN**.  
It combines a trained ResNet50 model, a user-friendly [Streamlit](https://streamlit.io/) app for image upload and explainability (Grad-CAM), and a robust statistical inference script for model evaluation.

---

## Features

- **Streamlit Web App**: Upload X-ray images, get instant predictions, and visualize Grad-CAM heatmaps for explainability.
- **Statistical Inference & Evaluation**: Detailed metrics (accuracy, precision, recall, F1-score), confusion matrix, and bootstrapped confidence intervals.
- **Easy-to-use**: Step-by-step instructions, clean codebase, and modular structure.

---

## Directory Structure

```
ChestXray-Classification/
│
├── app.py                           # Streamlit web app (frontend)
├── statistical_inference_evaluation.py   # Evaluation & statistical inference script
├── best_model.pth                   # Trained model weights (see note below)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .gitignore
│
├── test/                            # Test images (not uploaded; see below)
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   ├── TUBERCULOSIS/
│   └── UNKNOWN/
│
└── images/                          # (Optional) Example images/results for documentation
```

---

## Setup & Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/aadilshaikh123/ChestXray-Classification.git
   cd ChestXray-Classification
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Download/Place model weights**
   - Place `best_model.pth` in the root folder (not tracked in repo if large).

4. **Prepare the test set (for statistical inference)**
   - Place test images in the `test/` folder, structured as:
     ```
     test/
       NORMAL/
       PNEUMONIA/
       TUBERCULOSIS/
       UNKNOWN/
     ```
   - **Data is not included** due to privacy/size.

---

## How to Use

### 1. **Run the Streamlit App**

```sh
streamlit run app.py
```
- Open the provided local URL in your browser.
- Upload a chest X-ray image or use the camera.
- View predicted class, probability, and Grad-CAM visualization.

---

### 2. **Statistical Inference & Model Evaluation**

```sh
python statistical_inference_evaluation.py
```
- Evaluates on your `test/` folder.
- Prints accuracy, precision, recall, F1-score, macro metrics, per-class metrics, and **bootstrapped 95% confidence intervals**.
- Plots confusion matrix and metric bar charts.

---

## Example Results

- **Accuracy:** 0.92 (95% CI: [0.89, 0.95])
- **Confusion Matrix:**
  ![confusion matrix](images/confusion_matrix.png)
- **Grad-CAM Example:**
  ![gradcam](images/gradcam_example.png)

---

## Statistical Inference Explanation

Statistical inference provides **uncertainty quantification** for model evaluation.  
This project uses **bootstrapping** to compute 95% confidence intervals for accuracy, precision, and recall, allowing robust conclusions about model performance and reliability.

---

## Notes

- **Large files**: `best_model.pth` and `test/` are ignored in version control. Use [Git LFS](https://git-lfs.com/) or provide download links if sharing.
- **Data privacy**: No actual patient data is included in this repo.
- **Extensibility**: Add more classes, models, or explainability tools as needed.

---

## Credits

- Model architecture: ResNet50 (PyTorch)
- UI: Streamlit
- Metrics: scikit-learn
- Explainability: Grad-CAM
- Author: [aadilshaikh123](https://github.com/aadilshaikh123)

---

## License

[MIT](LICENSE) (Add a LICENSE file if desired)

---
