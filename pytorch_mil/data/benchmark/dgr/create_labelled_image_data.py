import os

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CSV_OUT_PATH = 'data/DGR/targets.csv'
RAW_DATA_DIR = 'data/DGR/raw'

# Load and parse labelled image metadata (only original train images are labelled)
metadata_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'metadata.csv'))
metadata_df = metadata_df[metadata_df['split'] == 'train']
metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_pth: os.path.join(RAW_DATA_DIR, img_pth))
metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth: os.path.join(RAW_DATA_DIR, img_pth))

# Load class dict (name -> rbg) and convert rgb values to binary
class_dict = pd.read_csv(os.path.join(RAW_DATA_DIR, 'class_dict.csv'))
class_rgbs = class_dict[['r', 'g', 'b']].values.tolist()
class_rgbs = [",".join([str(v//255) for v in class_rgb]) for class_rgb in class_rgbs]

# Add extra targets column to df
metadata_df['targets'] = [None] * len(metadata_df)

# Loop through every labelled image
for i in tqdm(range(len(metadata_df)), "Parsing targets"):
    # Load mask and convert it to binary
    mask_path = metadata_df['mask_path'][i]
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
    binary_mask = torch.zeros_like(torch.as_tensor(mask))
    binary_mask[mask > 128] = 1

    # Get unique pixel values that indicate different targets
    rgbs = np.unique(binary_mask.reshape(-1, binary_mask.shape[2]), axis=0)

    # Convert pixel values into target list
    targets = []
    for rgb in rgbs:
        target = class_rgbs.index(",".join([str(v) for v in rgb]))
        targets.append(target)

    # Write targets to df
    metadata_df.at[i, 'targets'] = ",".join([str(t) for t in sorted(targets)])

# Save image id and targets to new file
metadata_df.to_csv(CSV_OUT_PATH, index=False, columns=['image_id', 'targets'])
