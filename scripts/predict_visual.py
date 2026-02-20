import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scripts.model import BrainTumorModel

# Load trained model
model = BrainTumorModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Folder with processed .npz files
processed_dir = r"C:\Users\sgari\Downloads\aimodel\data\processed"
files = os.listdir(processed_dir)

# Pick 5 random patients
sample_files = random.sample(files, 10)

for f in sample_files:
    data_path = os.path.join(processed_dir, f)
    data = np.load(data_path)
    image = data['image']
    mask = data['mask']

    slice_idx = 60
    slice_img = image[:, slice_idx, :, :]
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-5)
    slice_tensor = torch.tensor(slice_img, dtype=torch.float32).unsqueeze(0)

    # Predict
    output = model(slice_tensor)
    pred = torch.argmax(output, dim=1).item()
    label_text = "Tumor" if pred == 1 else "No Tumor"
    print(f"{f}: Prediction -> {label_text}")

    # Visualize all 4 MRI modalities
    plt.figure(figsize=(12,4))
    modalities = ['FLAIR','T1','T1CE','T2']
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(slice_img[i], cmap='gray')
        plt.title(f"{modalities[i]}\n{label_text}")
        plt.axis('off')
    plt.show()
