import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.deepcfd.models.UNetEx import UNetEx
from data_utils.cfd_dataset import CaseDataDataset


def visualize_results(prediction, label, filename):
    # Create a plot with the reversed grayscale colormap so 0=white, 1=black
    data = torch.cat((prediction, label), dim=0)
    data = data.cpu().numpy()
    plt.figure()
    plt.imshow(data, cmap='gray_r', vmin=0, vmax=1)  # vmin/vmax set to 0-1 for proper color scaling
    plt.axis('off')  # Optional: turn off axes for a cleaner image
    
    # Save the figure to a TIFF file
    plt.savefig(filename, format='png')
    plt.close()  # Close the figure to free memory


def evaluate(model, dataloader, device):
    """
    Evaluate the model on the test set and return the average MSE loss.
    """
    model.eval()
    mse_loss_fn = nn.MSELoss(reduction='mean')
    total_loss = 0.
    total_samples = 0.

    with torch.no_grad():
        for filename, inputs, targets in dataloader:
            # Move input to device
            inputs = inputs.to(device)  # Shape: (B, 1, H, W)
            # Move each target modality to device and then concatenate along channel dim.
            target_pressure    = targets['pressure'].to(device)
            target_temperature = targets['temperature'].to(device)
            target_velocity    = targets['velocity'].to(device)
            
            # Concatenate targets: final shape (B, 3, H, W)
            targets = torch.cat([
                target_pressure,
                target_temperature,
                target_velocity], dim=1).to(device)
            
            # Forward pass
            outputs = model(inputs).squeeze()
            contour = inputs[0]
            pred_P, pred_T, pred_V = outputs[0], outputs[1], outputs[2]
            
            visualize_results(target_pressure.squeeze(), pred_P, filename=f'{filename}-pressure.png')
            visualize_results(target_temperature.squeeze(), pred_T, filename=f'{filename}-temperature.png')
            visualize_results(target_velocity.squeeze(), pred_V, filename=f'{filename}-velocity.png')
            
    #         loss = mse_loss_fn(outputs, targets)
    #         # Accumulate loss weighted by batch size
    #         batch_size = inputs.size(0)
    #         total_loss += loss.item() * batch_size
    #         total_samples += batch_size

    # avg_loss = total_loss / total_samples
    # return avg_loss


def main():
    # Test set root directory (should contain subfolders: contour, pressure, temperature, velocity)
    test_root_dir = 'data/case_data2/fluent_data_fig'
    batch_size = 1
    num_workers = 2

    # Build the test dataset and dataloader
    test_dataset = CaseDataDataset(test_root_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model (input: contour image with 1 channel, output: 3 channels for pressure, temperature, velocity)
    model = UNetEx(in_channels=1, out_channels=3).to(device)
    
    # Optionally load a trained checkpoint if available
    checkpoint_path = os.path.join("checkpoints", "epoch_500.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path} ...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found; evaluating an untrained model.")

    results_path = "results"
    os.makedirs(results_path, exist_ok=True)
    # Evaluate the model on the test set using MSE
    model.eval()
    mse_loss_fn = nn.MSELoss(reduction='mean')
    total_loss = 0.
    total_samples = 0.

    with torch.no_grad():
        for filename, inputs, targets in test_loader:
            # Move input to device
            inputs = inputs.to(device)  # Shape: (B, 1, H, W)
            # Move each target modality to device and then concatenate along channel dim.
            target_pressure    = targets['pressure'].to(device)
            target_temperature = targets['temperature'].to(device)
            target_velocity    = targets['velocity'].to(device)
            
            # Concatenate targets: final shape (B, 3, H, W)
            targets = torch.cat([
                target_pressure,
                target_temperature,
                target_velocity], dim=1).to(device)
            
            # Forward pass
            outputs = model(inputs).squeeze()
            contour = inputs[0]
            
            visualize_results(target_pressure.squeeze(), outputs[0], filename=f'{results_path}/{filename}-pressure.png')
            visualize_results(target_temperature.squeeze(), outputs[1], filename=f'{results_path}/{filename}-temperature.png')
            visualize_results(target_velocity.squeeze(), outputs[2], filename=f'{results_path}/{filename}-velocity.png')

if __name__ == '__main__':
    main()
