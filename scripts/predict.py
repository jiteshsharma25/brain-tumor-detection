import torch
import numpy as np
from scripts.model import BrainTumorModel
# Load your trained model
model = BrainTumorModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Load one sample from processed data
data_path = r"C:\Users\sgari\Downloads\aimodel\data\processed\BraTS20_Training_001.npz"
data = np.load(data_path)

image = data['image']
mask = data['mask']

# Pick middle slice (already normalized in dataset)
slice_idx = 60
slice_img = image[:, slice_idx, :, :]

# Normalize
slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-5)

# Convert to tensor
slice_tensor = torch.tensor(slice_img, dtype=torch.float32).unsqueeze(0)  # batch=1

# Predict
output = model(slice_tensor)
pred = torch.argmax(output, dim=1).item()

if pred == 1:
    print("Tumor detected")
else:
    print("No tumor")
