import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.deepcfd.models.UNetEx import UNetEx
from data_utils.cfd_dataset import CaseDataDataset


def evaluate(model, dataloader, device):
    """
    Evaluate the model on the test set and return the average MSE loss.
    """
    model.eval()
    mse_loss_fn = nn.MSELoss(reduction='mean')
    total_loss = 0.
    total_samples = 0.

    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move input to device
            inputs = inputs.to(device)  # Shape: (B, 1, H, W)
            
            # Move each target modality to device and then concatenate along channel dim.
            target_pressure    = targets['pressure'].to(device)
            target_temperature = targets['temperature'].to(device)
            target_velocity    = targets['velocity'].to(device)
            
            # Concatenate targets: final shape (B, 3, H, W)
            targets_tensor = torch.cat([target_pressure,
                                        target_temperature,
                                        target_velocity], dim=1)
            
            # Forward pass
            outputs = model(inputs)
            loss = mse_loss_fn(outputs, targets_tensor)
            
            # Accumulate loss weighted by batch size
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss


def main():
    # Test set root directory (should contain subfolders: contour, pressure, temperature, velocity)
    test_root_dir = 'data/case_data2/fluent_data_fig'
    batch_size = 4
    num_workers = 2

    # Build the test dataset and dataloader
    test_dataset = CaseDataDataset(test_root_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model (input: contour image with 1 channel, output: 3 channels for pressure, temperature, velocity)
    model = UNetEx(in_channels=1, out_channels=3).to(device)
    
    # Optionally load a trained checkpoint if available
    checkpoint_path = os.path.join("checkpoints", "model_final.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path} ...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found; evaluating an untrained model.")
    
    # Evaluate the model on the test set using MSE
    avg_mse = evaluate(model, test_loader, device)
    print(f"Average Mean Squared Error on test set: {avg_mse:.6f}")

if __name__ == '__main__':
    main()
