import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.deepcfd.models.UNetEx import UNetEx
from data_utils.cfd_dataset import CaseDataDataset


def main():
    # Hyperparameters (assumed same as your __main__.py)
    num_epochs = 100
    batch_size = 4
    learning_rate = 1e-4

    # Define the root directory where your four child folders are located.
    # The directory should contain subfolders: 'contour', 'pressure', 'temperature', 'velocity'.
    root_dir = 'data/case_data1/fluent_data_fig'
    
    # Create the dataset and dataloader
    dataset = CaseDataDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model.
    # Assuming UNetEx accepts parameters for number of input and output channels.
    # Here, input is a single-channel contour image and output has three channels (pressure, temperature, velocity).
    model = UNetEx(in_channels=1, out_channels=3).to(device)
    
    # Define the loss function and optimizer.
    criterion = nn.MSELoss()  # or another loss function if required
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create a folder to save checkpoints
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            # Move the input contour image to the device.
            inputs = inputs.to(device)  # shape: (B, 1, H, W)
            
            # Targets is a dictionary with keys 'pressure', 'temperature', 'velocity'
            # Move each target modality to the device.
            target_pressure    = targets['pressure'].to(device)
            target_temperature = targets['temperature'].to(device)
            target_velocity    = targets['velocity'].to(device)
            
            # Concatenate the targets along the channel dimension to form a (B, 3, H, W) tensor.
            targets_tensor = torch.cat(
                [target_pressure, target_temperature, target_velocity], dim=1
            )
            
            # Forward pass
            outputs = model(inputs)
            print(outputs.shape, targets_tensor.shape)
            loss = criterion(outputs, targets_tensor)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        
        # Optionally, save a checkpoint every 10 epochs.
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
    
    print("Training complete.")

if __name__ == '__main__':
    main()
