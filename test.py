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
from src.deepcfd.metrics import abs_relative_difference, \
    squared_relative_difference, delta1_acc
    

def apply_colors_to_array(x, mask):
    """
    Args:
        x: numpy array of shape (H, W), values in [0, 1]
    Returns:
        rgb: numpy array of shape (3, H, W), dtype=float32, RGB values in [0, 1]
    """
    # Get the colormap
    color_map = colormaps.get_cmap('Spectral')
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


def evaluate(mask, pred, label):
    """
    Evaluate the model on the test set
    """
    abs_relative_diff = abs_relative_difference(
        output=pred,
        target=label,
        valid_mask=mask
    )
    delta1_accuracy = delta1_acc(
        pred=pred,
        gt=label,
        valid_mask=mask
    )
    
    sq_relative_diff = squared_relative_difference(
        output=pred,
        target=label,
        valid_mask=mask
    )
    return {'abs_relative_diff': abs_relative_diff, 
            'delta1_accuracy': delta1_accuracy,
            'sq_relative_diff':sq_relative_diff,
            }


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

    sum_AbsRel = 0.
    sum_SqRel = 0.
    sum_delta1_acc = 0.
    num_inputs = len(test_dataset)
    
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
            masks = (inputs.squeeze()[0].detach() > 0.5).bool()
            
            eval_results = evaluate(masks, outputs[0], targets[0])
            sum_AbsRel += eval_results['abs_relative_diff']
            sum_SqRel += eval_results['sq_relative_diff']
            sum_delta1_acc += eval_results['delta1_accuracy']
            contour = masks.float()
            
            # base_path = f"{results_path}/{data_dict['filepath'][0]}"
            # visualize_results(
            #     contour, outputs[0], targets[0], filename=f'{base_path}-p.png')
            # visualize_results(
            #     contour, outputs[1], targets[1], filename=f'{base_path}-t.png')
            # visualize_results(
            #     contour, outputs[2], targets[2], filename=f'{base_path}-v.png')

        avg_AbsRel = sum_AbsRel / num_inputs
        avg_SqRel = sum_SqRel / num_inputs
        avg_del_acc = sum_delta1_acc / num_inputs
        
        print(f"Number of test inputs: {num_inputs}\n"
              f"Absolute Mean Relative Error (AbsRel): {avg_AbsRel:.4f}\n"
              f"Square Mean Relative Error (SqRel): {avg_SqRel:.4f}\n"
              f"delta1 Accuracy: {avg_del_acc:.4f}"
              )

if __name__ == '__main__':
    main()
