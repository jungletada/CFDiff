import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
 
FILES_TRAIN = 'list_train.txt'
FILES_TEST = 'list_test.txt'

STAT_pressure={'min': -37.73662186, 'max': 57.6361618}
STAT_temperature={'min': 299.9764404, 'max':310.3595276}
STAT_velocity={'min': 0.0, 'max':0.3930110071636349}


def visualize_field(input_image, filename, denormalize=True):
    """
    Visualize a 2D tensor or array in grayscale (0 -> white, 1 -> black) and save as a TIFF file.
    
    Parameters:
        field_tensor: The input data (NumPy array or PyTorch tensor) to visualize.
        filename:     The filename (string) for saving the image (should end with .tiff).
    """
    # If input is a PyTorch tensor, convert it to a NumPy array for Matplotlib
    try:
        if isinstance(input_image, torch.Tensor):
            data = input_image.detach().cpu().numpy()  # move to CPU and convert to numpy
        # elif isinstance(input_image, np.array):
        #     data = np.array(input_image)
    except ImportError:
        # If torch isn't available, just convert using NumPy
        data = np.array(input_image)
    
    # Remove any singleton dimensions (e.g., (1, H, W) -> (H, W))
    data = np.squeeze(data)
    
    if denormalize:
        data = (data + 1.) / 2. * 255.  
    
    # Create a plot with the reversed grayscale colormap so 0=white, 1=black
    plt.figure()
    plt.imshow(data, cmap='gray_r', vmin=0, vmax=255)  # vmin/vmax set to 0-1 for proper color scaling
    plt.axis('off')  # Optional: turn off axes for a cleaner image
    
    # Save the figure to a TIFF file
    plt.savefig(os.path.join('demo', filename), format='png')
    plt.close()  # Close the figure to free memory
    # Log the image to wandb


class CFDDataset(Dataset):
    def __init__(
        self,
        root_dir,
        is_train=True,
        ):
        
        self.data_root = root_dir
        self.is_train = is_train
        
        if self.is_train:
            list_file = FILES_TRAIN
        else:
            list_file = FILES_TEST
            
        self._suffix = '.png'
            
        self.contour_dir = os.path.join(root_dir, 'contour')
        self.pressure_dir = os.path.join(root_dir, 'pressure')
        self.temperature_dir = os.path.join(root_dir, 'temperature')
        self.velocity_dir = os.path.join(root_dir, 'velocity')
        

        self.filenames = [name.replace('.png', '') for name in os.listdir(self.contour_dir)]
        # self.filenames = [] 
        # with open(os.path.join(root_dir, list_file), 'r') as f:
        #     self.filenames = [line.strip() for line in f.readlines()]
        
        # self.filepaths = []
        # for attribute in ['pressure', 'temperature', 'velocity']:
        #     for filename in self.filenames:
        #         self.filepaths.append(os.path.join(attribute, filename))
                
        self._length = len(self.filenames)

    def __len__(self):
        return self._length

    def transform_image(self, image_path):
        image = Image.open(image_path + self._suffix).convert('L')
        image = torch.from_numpy(np.array(image))
        image = image.unsqueeze(0) # h, w -> 1, h, w
        image = image / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        return image
        
    def __getitem__(self, i):
        filepath_t = os.path.join(self.temperature_dir, self.filenames[i])
        filepath_p = os.path.join(self.pressure_dir,    self.filenames[i])
        filepath_v = os.path.join(self.velocity_dir,    self.filenames[i])
        filepath_c = os.path.join(self.contour_dir,     self.filenames[i])
        
        image_c = self.transform_image(filepath_c)
        image_p = self.transform_image(filepath_p)
        image_v = self.transform_image(filepath_v)
        image_t = self.transform_image(filepath_t)
      
        _, h, w = image_c.shape
        line_values = torch.linspace(1, 0, w, dtype=torch.float32)
        image_f = line_values.repeat(h, 1).unsqueeze(0) # h, w
        
        flip_flag = np.random.rand() < 0.5
        if flip_flag and self.is_train == 'train':
            image_c = image_c.flip(-1)
            image_p = image_p.flip(-1)
            image_t = image_t.flip(-1)
            image_v = image_v.flip(-1)
            image_f = image_f.flip(-1)
            
        example = {'inputs': torch.cat((image_c, image_f), dim=0),
                   'targets': torch.cat((image_p, image_t, image_v), dim=0),
                   'filepath': self.filenames[i]}
        
        return example
    
   
if __name__ == '__main__':
    root_dir = 'data/case_data1/fluent_data_map'  # Root folder containing the four child folders.
    dataset = CFDDataset(root_dir)
    
    # Create a DataLoader for batching.
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    os.makedirs('demo', exist_ok=True)
    
    # Iterate over one batch.
    for i, data_ in enumerate(dataloader):
        inputs, targets, filenames = data_['inputs'], data_['targets'], data_['filepath']
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
