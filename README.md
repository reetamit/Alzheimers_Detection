# 🧠 AI-Driven Alzheimer’s Stage Classification Using MRI and Deep Learning

## 🔍 Abstruct

Alzheimer’s disease affects millions globally, yet early-stage diagnosis remains challenging due to subtle anatomical changes and limited access to expert radiologists. This project builds a transparent, interpretable AI system that classifies Alzheimer’s stages (0–3) from MRI brain scans—empowering clinicians with reliable decision support and advancing ethical medical AI.

**Key Goals:**
- Early detection to improve patient outcomes  
- Model interpretability to build clinical trust  
- Fairness and transparency across demographic subgroups  
- Open-source reproducibility for global accessibility  

---

## 📦 Data Sources

The model is trained on publicly available **Kaggle Alzheimer’s MRI datasets**, structured into DataFrames with stage labels and metadata. All scans are anonymized and preprocessed for consistency.

---

## 🧪 Methods

### 🗂️ Data Processing
- **Loading & Preprocessing**: MRI images ingested from structured DataFrames  
- **Custom PyTorch Dataset**: Handles image-label pairing and metadata parsing  
- **Image Transformation**: Resizing, normalization, and optional Mixup augmentation  

### 🧠 Model Processing
- **Architecture**: EfficientNet backbone with dynamic flattening  
- **Class Imbalance Handling**: Weighted loss functions and stratified sampling  
- **Training Strategy**: Mixup augmentation, early stopping, and adaptive learning rate scheduling  

---

## 📈 Evaluation

### ✅ Validation Phase
- Early stopping logic  
- Batch-level performance tracking  
- Accuracy comparison across architectures  

### 📊 Performance Metrics
- **Test Accuracy**: 93.67%  
- **Loss Curves**: Stable convergence over 20 epochs  
- **Confusion Matrix**: High precision for Classes 2 and 3  
- **Entropy Plot**: Highlights uncertain predictions for human review  

---

## 🔍 Interpretability Tools

### 🔹 SHAP (SHapley Additive Explanations)
- Voxel-level feature attribution  
- Reveals regions influencing model predictions  
![SHAP Visualization](./data/shap_fig.png)

### 🔹 Grad-CAM Overlays
- Highlights spatial attention per class  
- Distinct activation patterns across stages  
![Grad-CAM Visualization](./data/grad.png)

### 🔹 Confidence Scores
- Per-sample prediction certainty across 4 classes  
![Confidence Score Chart](./data/cs.png)

### 🔹 Prediction Uncertainty
- Entropy-based visualization for 10 samples  
![Entropy Plot](./data/pu.png)

### 🔹 Confusion Matrix
- True vs predicted label distribution  
![Confusion Matrix](./data/cm.png)

---

## 🌍 Impact

This project blends technical rigor with ethical design, offering a reproducible, interpretable tool for Alzheimer’s detection.

---

## 🚀 How to Run

    ```bash
    # Clone the repo
    git clone https://github.com/your-username/alzheimer-mri-classification.git
    cd alzheimer-mri-classification

    # Install dependencies
    pip install -r requirements.txt

## License
    This project is licensed under
    GNU General Public License version 3 or later (GPL3+)