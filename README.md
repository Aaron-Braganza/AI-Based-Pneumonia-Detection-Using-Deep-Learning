# Pediatric Pneumonia Detection & Interpretability (ResNet-18)

An end-to-end deep learning pipeline for pediatric chest radiograph triage featuring custom Thoracic Anatomic Cropping, post-hoc threshold calibration, and intrapulmonary Grad-CAM explainability.

## Key Performance Metrics
- **ROC-AUC:** 0.9609
- **Sensitivity (Recall):** 98.21% (383 / 390 positive cases detected)
- **Calibrated Operating Threshold:** $\tau = 0.9752$ (capped false positives at $\le 60$)
- **Pneumonia Precision:** 86.46%
- **Overall Accuracy:** 89.26%

## Architecture & Pipeline
- **Backbone:** Fine-tuned ResNet-18 with transfer learning.
- **Preprocessing:** Custom Thoracic Anatomic Crop to strip scanner borders, neck artifacts, and subdiaphragmatic gas pockets.
- **Explainable AI (XAI):** Layer-hooked Grad-CAM generating real-time intrapulmonary saliency heatmaps.
- **Deployment:** Compiled TorchScript binary served via a lightweight local Gradio web UI.

## Local Setup & Inference
```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install gradio opencv-python pillow numpy
python app.py