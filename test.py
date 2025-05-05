import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.deepcfd.models.UNetEx import UNetEx
from data_utils.cfd_dataset import CFDDataset


def visualize_results(contour, pred, label, filename, denormalize=True):
    # Create a plot with the reversed grayscale colormap so 0=white, 1=black
    if denormalize:
        pred  = (pred + 1.) / 2. * 255.  
        label = (label + 1.) / 2. * 255.  
        contour[contour <=0.5] = 0
        contour[contour != 0] = 1
        
    plot_data = torch.cat((pred * contour, label * contour), dim=0)
    plot_data = plot_data.cpu().numpy()
    plt.figure()
    plt.imshow(
        plot_data, cmap='gray_r', vmin=0, vmax=255)  # vmin/vmax set to 0-1 for proper color scaling
    plt.axis('off')  # Optional: turn off axes for a cleaner image
    # Save the figure to a TIFF file
    plt.savefig(filename, format='png')
    plt.close()  # Close the figure to free memory


def evaluate(model, dataloader, device):
    """
    Evaluate the model on the test set and return the average MSE loss.
    """
    model.eval()
    with torch.no_grad():
        for filename, inputs, targets in dataloader:
            # Move input to device
            inputs = inputs.to(device)  # Shape: (1, 2, H, W)
            # Move each target modality to device and then concatenate along channel dim.
            target_pressure    = targets['pressure'].to(device)
            target_temperature = targets['temperature'].to(device)
            target_velocity    = targets['velocity'].to(device)
            
            # Forward pass
            outputs = model(inputs).squeeze()
            contour = inputs[0]
            pred_P, pred_T, pred_V = outputs[0], outputs[1], outputs[2]
            
            visualize_results(target_pressure.squeeze(), pred_P, filename=f'{filename}-pressure.png')
            visualize_results(target_temperature.squeeze(), pred_T, filename=f'{filename}-temperature.png')
            visualize_results(target_velocity.squeeze(), pred_V, filename=f'{filename}-velocity.png')


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
            label_p = targets[0]
            label_p = targets[1]
            label_v = targets[2]
            
            # Forward pass
            outputs = model(inputs).squeeze()
            contour = inputs.squeeze()[0]

            base_path = f"{results_path}/{data_dict['filepath'][0]}"
            visualize_results(
                contour, label_p, outputs[0], filename=f'{base_path}-pressure.png')
            visualize_results(
                contour, label_p, outputs[1], filename=f'{base_path}-temperature.png')
            visualize_results(
                contour, label_v, outputs[2], filename=f'{base_path}-velocity.png')

if __name__ == '__main__':
    main()
