import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
 
FILES_TRAIN = 'list_train.txt'
FILES_TEST = 'list_test.txt'

demo_dir = 'demo'

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
    plt.savefig(os.path.join(demo_dir, filename), format='png')
    plt.close()  # Close the figure to free memory
    # Log the image to wandb


# class CaseDataDataset(Dataset):
#     def __init__(self, root_dir, train=True):
#         """
#         Args:
#             root_dir (str): Root directory containing the four child folders:
#                 'contour', 'pressure', 'temperature', 'velocity'.
#             transform (callable, optional): Transform function to apply to each image.
#                 If None, a default resize transform is applied that sets the image's
#                 height to 256 while preserving its aspect ratio.
#         """
#         self.root_dir = root_dir
#         self.contour_dir = os.path.join(root_dir, 'contour')
#         self.pressure_dir = os.path.join(root_dir, 'pressure')
#         self.temperature_dir = os.path.join(root_dir, 'temperature')
#         self.velocity_dir = os.path.join(root_dir, 'velocity')
#         self.file_names = sorted(os.listdir(self.contour_dir))

#     def __len__(self):
#         return len(self.file_names)

#     def __getitem__(self, idx):
#         file_name = self.file_names[idx]
#         # Construct full paths for each modality.
#         contour_path     = os.path.join(self.contour_dir, file_name)
#         pressure_path    = os.path.join(self.pressure_dir, file_name)
#         temperature_path = os.path.join(self.temperature_dir, file_name)
#         velocity_path    = os.path.join(self.velocity_dir, file_name)
        
#         # Read and transform images.
#         contour_image     = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
#         pressure_image    = cv2.imread(pressure_path, cv2.IMREAD_GRAYSCALE)
#         temperature_image = cv2.imread(temperature_path, cv2.IMREAD_GRAYSCALE)
#         velocity_image    = cv2.imread(velocity_path, cv2.IMREAD_GRAYSCALE)

#         h, w = contour_image.shape
#         line_values = np.linspace(255, 0, w, dtype=np.float32) # generate a fluent velocity data.
#         flow_image = np.tile(line_values, (h, 1))
#         image_dict = {
#             "contour": contour_image,
#             "pressure": pressure_image,
#             "temperature": temperature_image,
#             "velocity": velocity_image,
#             "flow": flow_image
#         }
#         image_dict = self.transform(image_dict)

#         # The contour image is the input and the other three are the targets.
#         input_tensor = torch.cat(
#             (image_dict['contour'],image_dict['flow']), dim=0)
#         target_tensor = torch.cat(
#             (image_dict['pressure'], image_dict['temperature'], image_dict[velocity]), dim=0)
#         name = file_name.replace('.png', '')
#         return name, input_tensor, target_tensor


class CFDDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode='train',
        ):
        
        self.data_root = root_dir
        self.mode = mode
        
        if self.mode == 'train':
            list_file = FILES_TRAIN
        else:
            list_file = FILES_TEST
            
        self._suffix = '.png'
            
        self.contour_dir = os.path.join(root_dir, 'contour')
        self.pressure_dir = os.path.join(root_dir, 'pressure')
        self.temperature_dir = os.path.join(root_dir, 'temperature')
        self.velocity_dir = os.path.join(root_dir, 'velocity')
        
        self.filenames = [] 
        with open(os.path.join(root_dir, list_file), 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]
        
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
        if flip_flag and self.mode == 'train':
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

    os.makedirs(demo_dir, exist_ok=True)
    
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
