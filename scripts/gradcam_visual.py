import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import random
from model import BrainTumorModel
from collections import OrderedDict

# -----------------------------
# Load model
# -----------------------------
model = BrainTumorModel()

# ---- Fix state_dict mismatch if needed ----
state_dict = torch.load("saved_models/model.pth")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace('conv.0', 'conv1').replace('conv.3', 'conv2').replace('conv.6', 'conv3')
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)
model.eval()

# -----------------------------
# Processed data
# -----------------------------
processed_dir = r"data/processed"
files = os.listdir(processed_dir)
sample_files = random.sample(files, 3)

os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Grad-CAM + Visualization
# -----------------------------
for f in sample_files:
    data = np.load(os.path.join(processed_dir, f))
    image = data['image']       # [4, slices, H, W]
    mask = data['mask']         # segmentation mask [slices, H, W]

    # Pick slice with largest tumor
    slice_idx = np.argmax(mask.sum(axis=(1,2)))
    slice_img = image[:, slice_idx, :, :]
    slice_mask = mask[slice_idx]

    # Normalize MRI
    slice_img_norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
    slice_tensor = torch.tensor(slice_img_norm, dtype=torch.float32).unsqueeze(0)

    # Forward pass
    output = model(slice_tensor)
    pred = torch.argmax(output, dim=1).item()
    label = "Tumor" if pred == 1 else "No Tumor"
    print(f"{f}: Prediction -> {label}")

    # -----------------------------
    # Grad-CAM
    # -----------------------------
    model.zero_grad()
    output[0, pred].backward()

    gradients = model.get_activations_gradient()
    activations = model.get_activations(slice_tensor).detach()
    pooled_gradients = torch.mean(gradients, dim=[0,2,3])
    for i in range(activations.shape[1]):
        activations[0,i,:,:] *= pooled_gradients[i]

    heatmap = torch.mean(activations[0], dim=0)
    heatmap = F.relu(heatmap)
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-8  # normalize 0-1

    # -----------------------------
    # Plot all 4 modalities
    # -----------------------------
    modalities = ['FLAIR','T1','T1CE','T2']
    plt.figure(figsize=(16,4))
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(slice_img_norm[i], cmap='gray')
        plt.imshow(heatmap.cpu().numpy(), cmap='jet', alpha=0.5)       # Grad-CAM
        plt.contour(slice_mask, colors='lime', linewidths=1.0)        # tumor outline
        plt.title(f"{modalities[i]}\n{label}")
        plt.axis('off')

    # Add legend/disclaimer
    tumor_patch = mpatches.Patch(color='lime', label='Tumor region')
    plt.legend(handles=[tumor_patch], loc='lower right')

    # Save figure
    plt.savefig(f"outputs/{f}_slice{slice_idx}_gradcam.png")
    plt.show()
