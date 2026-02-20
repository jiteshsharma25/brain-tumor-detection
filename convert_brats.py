import os
import nibabel as nib
import numpy as np
import cv2
import pandas as pd

# 👉 CHANGE THIS PATH TO YOUR ACTUAL DATA
data_dir = r"C:\Users\sgari\Downloads\aimodel\data\BraTS2021\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

# Output folders
os.makedirs("data/processed/images", exist_ok=True)
os.makedirs("data/processed/masks", exist_ok=True)

labels = []

count = 0

for case in os.listdir(data_dir):

    case_path = os.path.join(data_dir, case)

    if not os.path.isdir(case_path):
        continue

    print("Processing:", case)

    try:
        flair_path = os.path.join(case_path, f"{case}_flair.nii")
        seg_path = os.path.join(case_path, f"{case}_seg.nii")

        if not os.path.exists(flair_path) or not os.path.exists(seg_path):
            print("Missing files, skipping")
            continue

        flair = nib.load(flair_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()

        # middle slice
        z = flair.shape[2] // 2

        img = flair[:, :, z]
        mask = seg[:, :, z]

        # normalize image
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)

        # resize
        img = cv2.resize(img, (128, 128))
        mask = cv2.resize(mask, (128, 128))

        # label
        label = 1 if np.sum(mask) > 0 else 0

        # save image
        img_path = f"data/processed/images/{count}.png"
        mask_path = f"data/processed/masks/{count}.png"

        cv2.imwrite(img_path, img)
        cv2.imwrite(mask_path, mask)

        labels.append([f"{count}.png", label])

        count += 1

    except Exception as e:
        print("Error:", e)

print("Total images saved:", count)

# save labels
df = pd.DataFrame(labels, columns=["image", "label"])
df.to_csv("data/processed/labels.csv", index=False)

print("DONE")