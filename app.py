import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import nibabel as nib
import tempfile

# -------------------------------
# MODEL (MATCH TRAINED MODEL)
# -------------------------------
class BrainTumorModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -------------------------------
# LOAD MODEL
# -------------------------------
model = BrainTumorModel()
model.load_state_dict(torch.load("saved_models/model.pth", map_location="cpu"))
model.eval()

# -------------------------------
# LOAD NII
# -------------------------------
def load_nii(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
        tmp.write(file.read())
        path = tmp.name
    return nib.load(path).get_fdata()

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(flair, t1, t1ce, t2):
    z = flair.shape[2] // 2

    flair = flair[:, :, z]
    t1 = t1[:, :, z]
    t1ce = t1ce[:, :, z]
    t2 = t2[:, :, z]

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    flair_n = norm(flair)
    t1_n = norm(t1)
    t1ce_n = norm(t1ce)
    t2_n = norm(t2)

    flair_n = cv2.resize(flair_n, (128, 128))
    t1_n = cv2.resize(t1_n, (128, 128))
    t1ce_n = cv2.resize(t1ce_n, (128, 128))
    t2_n = cv2.resize(t2_n, (128, 128))

    img = np.stack([flair_n, t1_n, t1ce_n, t2_n], axis=0)
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return tensor, flair_n

# -------------------------------
# IMPROVED GRAD-CAM
# -------------------------------
def gradcam(model, input_tensor):
    gradients = []
    activations = []

    def f_hook(module, inp, out):
        activations.append(out)

    def b_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.conv[6]
    target_layer.register_forward_hook(f_hook)
    target_layer.register_full_backward_hook(b_hook)

    output = model(input_tensor)
    pred = torch.argmax(output)

    model.zero_grad()
    output[0, pred].backward()

    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]

    weights = np.mean(grads, axis=(1,2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (128,128))

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    return cam

# -------------------------------
# UI CONFIG (PROFESSIONAL)
# -------------------------------
st.set_page_config(page_title="AI Tumor Diagnosis", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;color:#00C853;'>🧠 AI Tumor Diagnosis System</h1>
<p style='text-align:center;color:gray;'>Deep Learning Assisted Medical Imaging</p>
<hr>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Patient Information")

gender = st.sidebar.selectbox("Gender", ["Select", "Male", "Female"])
age = st.sidebar.number_input("Age", 1, 120, 25)

# -------------------------------
# UPLOAD
# -------------------------------
st.subheader("Upload MRI Modalities (.nii)")

c1, c2, c3, c4 = st.columns(4)

with c1:
    flair_file = st.file_uploader("FLAIR", type=["nii"])
with c2:
    t1_file = st.file_uploader("T1", type=["nii"])
with c3:
    t1ce_file = st.file_uploader("T1CE", type=["nii"])
with c4:
    t2_file = st.file_uploader("T2", type=["nii"])

# -------------------------------
# MAIN
# -------------------------------
if flair_file and t1_file and t1ce_file and t2_file:

    flair = load_nii(flair_file)
    t1 = load_nii(t1_file)
    t1ce = load_nii(t1ce_file)
    t2 = load_nii(t2_file)

    input_tensor, display_img = preprocess(flair, t1, t1ce, t2)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob).item()
        confidence = prob[0][pred].item() * 100

    label = "Tumor Detected" if pred == 1 else "No Tumor"

    heatmap = gradcam(model, input_tensor)

    img = (display_img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # -------------------------------
    # BETTER SEGMENTATION (FIX)
    # -------------------------------
    thresh = np.percentile(heatmap, 90)  # dynamic threshold
    mask = (heatmap > thresh).astype(np.uint8)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    seg = img.copy()
    cv2.drawContours(seg, contours, -1, (0,255,0), 2)

    # heatmap overlay
    heatmap_color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    # -------------------------------
    # DISPLAY
    # -------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("MRI Slice")
        st.image(img, use_column_width=True)

    with col2:
        st.subheader("AI Attention Map")
        st.image(overlay, use_column_width=True)

    with col3:
        st.subheader("Detected Tumor Region")
        st.image(seg, use_column_width=True)

    # -------------------------------
    # REPORT
    # -------------------------------
    st.markdown("---")
    st.subheader("AI Diagnosis Report")

    if pred == 1:
        st.error(f"Tumor Detected | Confidence: {confidence:.2f}%")
    else:
        st.success(f"No Tumor Detected | Confidence: {confidence:.2f}%")

    st.info(f"Patient: Age {age}, Gender {gender}")

    st.warning("This is an AI-based analysis. Not a clinical diagnosis.")