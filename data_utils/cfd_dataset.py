import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


def visualize_field(field_tensor, filename):
    """
    Visualize a 2D tensor or array in grayscale (0 -> white, 1 -> black) and save as a TIFF file.
    
    Parameters:
        field_tensor: The input data (NumPy array or PyTorch tensor) to visualize.
        filename:     The filename (string) for saving the image (should end with .tiff).
    """
    # If input is a PyTorch tensor, convert it to a NumPy array for Matplotlib
    try:
        if isinstance(field_tensor, torch.Tensor):
            data = field_tensor.detach().cpu().numpy()  # move to CPU and convert to numpy
        else:
            data = np.array(field_tensor)
    except ImportError:
        # If torch isn't available, just convert using NumPy
        data = np.array(field_tensor)
    
    # Remove any singleton dimensions (e.g., (1, H, W) -> (H, W))
    data = np.squeeze(data)
    
    # Create a plot with the reversed grayscale colormap so 0=white, 1=black
    plt.figure()
    plt.imshow(data, cmap='gray_r', vmin=0, vmax=1)  # vmin/vmax set to 0-1 for proper color scaling
    plt.axis('off')  # Optional: turn off axes for a cleaner image
    
    # Save the figure to a TIFF file
    plt.savefig(filename, format='tiff')
    plt.close()  # Close the figure to free memory


def default_resize_transform(image, target_height=256, target_width=1024):
    """
    Resizes a single-channel image (a NumPy array) so that its height is target_height,
    while preserving the aspect ratio using linear interpolation.
    
    Args:
        image (np.ndarray): Input single-channel image.
        target_height (int): Desired height after resizing.
        
    Returns:
        np.ndarray: Resized image.
    """
    original_height, original_width = image.shape
    scale = target_height / original_height
    new_width = int(original_width * scale)
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
    width_left = (new_width - target_width) // 2
    resized = resized[:, width_left:width_left+target_width]
    return resized


class CaseDataDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing the four child folders:
                'contour', 'pressure', 'temperature', 'velocity'.
            transform (callable, optional): Transform function to apply to each image.
                If None, a default resize transform is applied that sets the image's
                height to 256 while preserving its aspect ratio.
        """
        self.root_dir = root_dir
        self.contour_dir = os.path.join(root_dir, 'contour')
        self.pressure_dir = os.path.join(root_dir, 'pressure')
        self.temperature_dir = os.path.join(root_dir, 'temperature')
        self.velocity_dir = os.path.join(root_dir, 'velocity')
        
        # Assume file names are identical across subfolders.
        self.file_names = sorted(os.listdir(self.contour_dir))
        
        # Use provided transform or the default resize transform.
        self.transform = transform if transform is not None else default_resize_transform

    def read_and_transform(self, path):
        # Read image in grayscale mode using OpenCV.
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        # Apply the transform to resize the image (shorter side height set to 256).
        img = self.transform(img)
        # Normalize: convert from 0-255 to 0-1, then invert (white becomes 0.0, black becomes 1.0).
        img = 1.0 - (img.astype(np.float32) / 255.0)
        # Add a channel dimension: (H, W) -> (1, H, W).
        img = np.expand_dims(img, axis=0)
        # Convert to torch tensor.
        tensor = torch.from_numpy(img)
        return tensor

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        # Construct full paths for each modality.
        contour_path     = os.path.join(self.contour_dir, file_name)
        pressure_path    = os.path.join(self.pressure_dir, file_name)
        temperature_path = os.path.join(self.temperature_dir, file_name)
        velocity_path    = os.path.join(self.velocity_dir, file_name)
        
        # Read and transform images.
        contour_tensor     = self.read_and_transform(contour_path)
        pressure_tensor    = self.read_and_transform(pressure_path)
        temperature_tensor = self.read_and_transform(temperature_path)
        velocity_tensor    = self.read_and_transform(velocity_path)
        
        # The contour image is the input and the other three are the targets.
        target = {
            'pressure': pressure_tensor,
            'temperature': temperature_tensor,
            'velocity': velocity_tensor
        }
        
        return contour_tensor, target


if __name__ == '__main__':
    root_dir = 'data/case_data1/fluent_data_fig'  # Root folder containing the four child folders.
    dataset = CaseDataDataset(root_dir)
    
    # Create a DataLoader for batching.
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Iterate over one batch.
    for inputs, targets in dataloader:
        print("Contour batch shape:", inputs.shape)
        print("Pressure batch shape:", targets['pressure'].shape)
        print("Temperature batch shape:", targets['temperature'].shape)
        print("Velocity batch shape:", targets['velocity'].shape)

        # Example: visualize a sample's input contour and target fields
        # (Assuming dataset[i] returns a tuple or dict: (contour, pressure, temperature, velocity))
        contour, pressure, temperature, velocity = inputs[0], targets['pressure'][0], targets['temperature'][0], targets['velocity'][0]

        visualize_field(contour, "sample_contour.tiff")
        visualize_field(pressure, "sample_pressure.tiff")
        visualize_field(temperature, "sample_temperature.tiff")
        visualize_field(velocity, "sample_velocity.tiff")

        break
