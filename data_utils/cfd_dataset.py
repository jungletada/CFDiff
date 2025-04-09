import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


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
    plt.savefig(filename, format='png')
    plt.close()  # Close the figure to free memory
    # Log the image to wandb


def transform_train(images, target_height=256, target_width=512):
    """
    Applies a random rotation (±10°) before resizing, random horizontal flip, and center cropping.
    If the input is a tuple or list of images, the same transformation is applied to each image.
    
    Args:
        images (np.ndarray or tuple/list of np.ndarray): Input single-channel image or tuple/list of images.
        target_height (int): Desired height after resizing.
        target_width (int): Desired width after cropping.
        
    Returns:
        np.ndarray or tuple: Transformed image or tuple of transformed images.
    """

    first_img = images[0]
    
    # flow_img = np.expand_dims(flow_img, axis=-1)
    # Apply random rotation on the first image to determine the common transformation parameters.
    angle = np.random.uniform(-15, 15)
    center = (first_img.shape[1] // 2, first_img.shape[0] // 2)  # (x, y) center
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Generate a random horizontal flip flag (True with 50% probability).
    flip_flag = np.random.rand() < 0.5
    
    # Apply rotation to the first image to compute the new dimensions (should remain same as original for cv2.warpAffine)
    # Use cv2.BORDER_REFLECT to avoid black borders.
    def transform_single(image):
        # 1. Random Rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 2. Resize the rotated image so that its height is target_height while preserving aspect ratio.
        original_height, original_width = rotated.shape
        scale = target_height / original_height
        new_width = int(original_width * scale)
        resized = cv2.resize(rotated, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        # 3. Random horizontal flip (50% chance) using the same decision for all images.
        # (We generate the flip flag once outside, so here we assume that variable is defined)
        if flip_flag:
            resized = np.fliplr(resized)
        
        # 4. Center crop to target_width.
        width_left = (resized.shape[1] - target_width) // 2
        cropped = resized[:, width_left:width_left + target_width]
        
        return cropped
    
    transformed_images = []
    for img in images:
        img = transform_single(img)
        img = 1.0 - (img.astype(np.float32) / 255.0)
        img = np.expand_dims(img, axis=0)
        tensor = torch.from_numpy(img)
        transformed_images.append(tensor)
    return transformed_images


def transform_test(images, target_height=256, target_width=512):
    """
    Applies a random rotation (±10°) before resizing, random horizontal flip, and center cropping.
    If the input is a tuple or list of images, the same transformation is applied to each image.
    
    Args:
        images (np.ndarray or tuple/list of np.ndarray): Input single-channel image or tuple/list of images.
        target_height (int): Desired height after resizing.
        target_width (int): Desired width after cropping.
        
    Returns:
        np.ndarray or tuple: Transformed image or tuple of transformed images.
    """

    # first_img = images[0]
    
    # flow_img = np.expand_dims(flow_img, axis=-1)
    # Apply random rotation on the first image to determine the common transformation parameters.
    # angle = np.random.uniform(-15, 15)
    # center = (first_img.shape[1] // 2, first_img.shape[0] // 2)  # (x, y) center
    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation to the first image to compute the new dimensions (should remain same as original for cv2.warpAffine)
    # Use cv2.BORDER_REFLECT to avoid black borders.
    def transform_single(image):
        # # 1. Random Rotation
        # rotated = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
        #                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 2. Resize the rotated image so that its height is target_height while preserving aspect ratio.
        original_height, original_width = image.shape
        scale = target_height / original_height
        new_width = int(original_width * scale)
        resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        # 3. Random horizontal flip (50% chance) using the same decision for all images.
        # (We generate the flip flag once outside, so here we assume that variable is defined)
        # if flip_flag:
        #     resized = np.fliplr(resized)
        # 4. Center crop to target_width.
        width_left = (resized.shape[1] - target_width) // 2
        cropped = resized[:, width_left:width_left + target_width]
        return cropped

    # Generate a random horizontal flip flag (True with 50% probability).
    # flip_flag = np.random.rand() < 0.5
    
    transformed_images = []
    for img in images:
        img = transform_single(img)
        img = 1.0 - (img.astype(np.float32) / 255.0)
        img = np.expand_dims(img, axis=0)
        tensor = torch.from_numpy(img)
        transformed_images.append(tensor)
    return transformed_images


class CaseDataDataset(Dataset):
    def __init__(self, root_dir, train=True):
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
        self.transform = transform_train if train else transform_test

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
        contour_image     = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        pressure_image    = cv2.imread(pressure_path, cv2.IMREAD_GRAYSCALE)
        temperature_image = cv2.imread(temperature_path, cv2.IMREAD_GRAYSCALE)
        velocity_image    = cv2.imread(velocity_path, cv2.IMREAD_GRAYSCALE)

        h, w = contour_image.shape
        line_values = np.linspace(255, 0, w, dtype=np.float32) # generate a fluent velocity data.
        flow_image = np.tile(line_values, (h, 1))
        tuple_images = (flow_image, contour_image, pressure_image, temperature_image, velocity_image)
        flow_tensor, contour_tensor, pressure_tensor, temperature_tensor, velocity_tensor = self.transform(tuple_images)

        # The contour image is the input and the other three are the targets.
        input_tensor = torch.cat((contour_tensor, flow_tensor), dim=0)
        target_tensor = torch.cat((pressure_tensor,temperature_tensor,velocity_tensor), dim=0)
        name = file_name.replace('s.tiff', '')
        return name, input_tensor, target_tensor


if __name__ == '__main__':
    root_dir = 'data/case_data1/fluent_data_fig'  # Root folder containing the four child folders.
    dataset = CaseDataDataset(root_dir)
    
    # Create a DataLoader for batching.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # Iterate over one batch.
    for filenames, inputs, targets in dataloader:
        print("Contour batch shape:", inputs.shape)
        print("Target batch shape:", targets.shape)
       
        # Example: visualize a sample's input contour and target fields
        # (Assuming dataset[i] returns a tuple or dict: (contour, pressure, temperature, velocity))
        index = 0
        filename = filenames[index]

        contour, flow = inputs[index][0], inputs[index][1]
        pressure, temperature, velocity = targets[index][0], targets[index][1], targets[index][2]
        
        visualize_field(flow, f"{filename}_flow.png")
        visualize_field(contour, f"{filename}_contour.png")
        visualize_field(pressure, f"{filename}_pressure.png")
        visualize_field(temperature, f"{filename}_temperature.png")
        visualize_field(velocity, f"{filename}_velocity.png")

        break
