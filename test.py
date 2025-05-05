import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import colormaps

from src.deepcfd.models.UNetEx import UNetEx
from data_utils.cfd_dataset import CFDDataset


def apply_colors_to_array(x, mask):
    """
    Args:
        x: numpy array of shape (H, W), values in [0, 1]
    Returns:
        rgb: numpy array of shape (3, H, W), dtype=float32, RGB values in [0, 1]
    """
    # Get the viridis colormap
    color_map = colormaps.get_cmap('viridis')
    # Apply the colormap (returns RGBA)
    rgba = color_map(x)  # shape: (H, W, 4)
    # Drop alpha and transpose to (3, H, W)
    rgb = rgba[..., :3]  # (3, H, W)
    rgb[mask == 0] = 1
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    return rgb_uint8
  

def visualize_results(mask, pred, label, filename):
    # Create a plot with the reversed grayscale colormap so 0=white, 1=black
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    mask = mask.cpu().numpy()
    
    pred_uint8 = apply_colors_to_array(x=pred, mask=mask)
    img = Image.fromarray(pred_uint8)
    img.save(filename)
    
    label_uint8 = apply_colors_to_array(x=label, mask=mask)
    img = Image.fromarray(label_uint8)
    img.save(filename.replace('.png', '_gt.png'))


def evaluate(mask, label, pred):
    """
    Evaluate the model on the test set
    """
    pass


def main():
    batch_size = 1
    num_workers = 2
    # Test set root directory (should contain subfolders: contour, pressure, temperature, velocity)
    test_root_dir = 'data/case_data2/fluent_data_map'
    # Build the test dataset and dataloader
    test_dataset = CFDDataset(
        test_root_dir, 
        mode='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers)
    
    # Set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model (input: contour image with 1 channel, output: 3 channels for pressure, temperature, velocity)
    model = UNetEx(in_channels=2, out_channels=3).to(device)
    
    # Optionally load a trained checkpoint if available
    checkpoint_path = os.path.join("checkpoints", "epoch_3000.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path} ...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found; evaluating an untrained model.")

    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    # Evaluate the model on the test set using MSE
    model.eval()

    with torch.no_grad():
        for data_dict in tqdm(test_loader):
            # Move input to device
            inputs = data_dict['inputs'].to(device)  # (B, 1, H, W)
            # Move each target modality to device and then concatenate along channel dim.
            targets = data_dict['targets'].squeeze().to(device)  # (3, H, W)
            targets = (targets + 1.) / 2. 
            
            # Forward pass
            outputs = model(inputs).squeeze().clip(-1.0, 1.0)
            outputs = (outputs + 1.) / 2. 
            
            # make mask
            masks = (inputs.squeeze()[0].detach() > 0.5)
            contour = masks.float()
        
            base_path = f"{results_path}/{data_dict['filepath'][0]}"
            visualize_results(
                contour, targets[0], outputs[0], filename=f'{base_path}-p.png')
            visualize_results(
                contour, targets[1], outputs[1], filename=f'{base_path}-t.png')
            visualize_results(
                contour, targets[2], outputs[2], filename=f'{base_path}-v.png')
            
            # masks = (inputs.squeeze()[0].detach() > 0.5)
            

if __name__ == '__main__':
    main()
