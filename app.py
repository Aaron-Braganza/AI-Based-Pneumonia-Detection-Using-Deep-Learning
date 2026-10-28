import os
import torch
import numpy as np
import cv2
from PIL import Image
import gradio as gr
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "pneumonia_retrained_model.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"'{MODEL_PATH}' was not found in the current directory. "
    )

print(f"Loading TorchScript model on device: {device}...")
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()

class ThoracicLungCrop:
    def __init__(self, crop_top=0.10, crop_bottom=0.20, crop_sides=0.12):
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_sides = crop_sides

    def __call__(self, img):
        w, h = img.size
        left = int(w * self.crop_sides)
        right = int(w * (1 - self.crop_sides))
        top = int(h * self.crop_top)
        bottom = int(h * (1 - self.crop_bottom))
        return img.crop((left, top, right, bottom))

crop_preprocess = ThoracicLungCrop(crop_top=0.10, crop_bottom=0.20, crop_sides=0.12)

input_transforms = transforms.Compose([
    crop_preprocess,
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

display_transforms = transforms.Compose([
    crop_preprocess,
    transforms.Resize((224, 224))
])

def generate_saliency_overlay(input_tensor, cropped_pil_img):
    """
    Computes input gradient attribution on the TorchScript module
    to generate an intrapulmonary activation heatmap.
    """
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    logits = model(input_tensor)
    
    # Backpropagate target pneumonia score (index 1)
    target_score = logits[0, 1]
    target_score.backward()
    
    # Extract gradient magnitude across channels
    grads = input_tensor.grad.squeeze().cpu().detach().numpy()
    cam = np.mean(np.maximum(grads, 0), axis=0)
    
    # Smooth with Gaussian blur and normalize to [0, 1]
    cam = cv2.GaussianBlur(cam, (11, 11), 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (224, 224))
    
    # Create colormap overlay
    orig_np = np.array(cropped_pil_img) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.6 * orig_np + 0.4 * heatmap, 0, 1)
    
    return Image.fromarray((overlay * 255).astype(np.uint8))

CALIBRATED_THRESHOLD = 0.9752

def diagnose_radiograph(input_image):
    if input_image is None:
        return "Please upload an image.", None, None

    pil_img = Image.fromarray(input_image).convert("RGB")
    cropped_display = display_transforms(pil_img)
    tensor = input_transforms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob_pneumonia = torch.softmax(logits, dim=1)[0, 1].item()
        prob_normal = 1.0 - prob_pneumonia

    is_pneumonia = prob_pneumonia >= CALIBRATED_THRESHOLD
    verdict = "PNEUMONIA DETECTED" if is_pneumonia else "NORMAL (CLEAR RADIOGRAPH)"

    results = {
        "Clinical Verdict": verdict,
        "Pneumonia Confidence": f"{prob_pneumonia * 100:.2f}%",
        "Normal Confidence": f"{prob_normal * 100:.2f}%",
        "Calibrated Operating Threshold": f"{CALIBRATED_THRESHOLD:.4f}"
    }

    # Generate explainability overlay if positive; return cropped view if normal
    if is_pneumonia:
        xai_image = generate_saliency_overlay(tensor, cropped_display)
    else:
        xai_image = cropped_display

    return verdict, results, xai_image

demo = gr.Interface(
    fn=diagnose_radiograph,
    inputs=gr.Image(label="Upload Pediatric Chest Radiograph (JPEG / PNG)"),
    outputs=[
        gr.Textbox(label="Diagnosis", lines=1),
        gr.JSON(label="Detailed Confidence Metrics"),
        gr.Image(label="Explainable AI: Pathological Attention (Cropped Lung Field)")
    ],
    title="Pediatric Chest X-Ray Screening System",
    description="Automated Pnemonia using fine-tuned ResNet-18 with Thoracic Anatomic Cropping Model Trained on Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification from https://data.mendeley.com/datasets/rscbjbr9sj/2.",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)