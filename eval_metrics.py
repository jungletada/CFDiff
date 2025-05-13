import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image


from matplotlib import colormaps
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


STAT_pressure={'min': -37.73662186, 'max': 57.6361618}
STAT_temperature={'min': 299.9764404, 'max':310.3595276}
STAT_velocity={'min': 0.0, 'max':0.3930110071636349}


def load_scale(key):
    if key.__contains__('pressure'):
        return STAT_pressure
    elif key.__contains__('temperature'):
        return STAT_temperature
    elif key.__contains__('velocity'):
        return STAT_velocity
    else:
        raise NotImplementedError
    
    
def apply_colors_to_array(x, mask, cmap='Spectral'):
    """
    Args:
        x: numpy array of shape (H, W), values in [0, 1]
    Returns:
        rgb: numpy array of shape (3, H, W), dtype=float32, RGB values in [0, 1]
    """
    # Get the colormap
    color_map = colormaps.get_cmap(cmap)
    # Apply the colormap (returns RGBA)
    rgba = color_map(x)  # shape: (H, W, 4)
    # Drop alpha and transpose to (3, H, W)
    rgb = rgba[..., :3]  # (3, H, W)
    if mask is not None:
        rgb[mask == 0] = 1
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    return rgb_uint8


def visualize_results(mask, pred, label, filename):
    # Create a plot with the reversed grayscale colormap so 0=white, 1=black
    # pred = pred.cpu().numpy()
    # label = label.cpu().numpy()
    # mask = mask.cpu().numpy()
    
    pred_uint8 = apply_colors_to_array(x=pred, mask=mask)
    image = Image.fromarray(pred_uint8)
    image.save(filename)
    
    # label_uint8 = apply_colors_to_array(x=label, mask=mask)
    # img = Image.fromarray(label_uint8)
    # img.save(filename.replace('.png', '_gt.png'))


def evaluate(pred, label, field, mask, denormalize=False):
    if len(mask.shape) == 4 or len(mask.shape) == 3:
        mask = mask.squeeze()
        pred = pred.squeeze()
        
    # mask = mask.cpu().numpy()
    # pred = pred.cpu().numpy()
    # label = label.cpu().numpy()
    
    if denormalize:
        stat = load_scale(field)
        pred = pred * (stat['max'] - stat['min']) + stat['min']
        label = label * (stat['max'] - stat['min']) + stat['min']

    img_true = (label * mask * 255.).astype(np.uint8)
    img_pred = (pred * mask * 255.).astype(np.uint8)
    # Flatten arrays to 1D for metric calculations
    y_true = label[mask].flatten()
    y_pred = pred[mask].flatten()
    
    # Compute metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)
    ssim_value = ssim(img_true, img_pred, data_range=img_true.max() - img_true.min())
    psnr_value = psnr(img_true, img_pred, data_range=img_true.max() - img_true.min())

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'SSIM': ssim_value,
        'PSNR': psnr_value
    }


class Evaluator:
    def __init__(self, num_samples):
        self.metrics = ['MAE', 'RMSE', 'R2', 'SSIM', 'PSNR']
        self.fields = ['pressure', 'temperature', 'velocity']
        self.sum_results = {field: {metric: 0. for metric in self.metrics} 
                            for field in self.fields}
        self.avg_results = {field: {metric: 0. for metric in self.metrics} 
                            for field in self.fields}
        self.denormalize = False
        self.num_samples = num_samples
        
    def evaluate_single(self, pred, label, field, mask):
        res = evaluate(pred, label, field, mask, denormalize=self.denormalize)
        
        for metric, value in res.items():
            self.sum_results[field][metric] += value
        return res
    
    def visualize_single(self, pred, filename, mask, label=None, flow=None):
        pred_uint8 = apply_colors_to_array(x=pred, mask=mask)
        image = Image.fromarray(pred_uint8)
        image.save(filename)
        
        if label is not None:
            label_uint8 = apply_colors_to_array(x=label, mask=mask)
            img = Image.fromarray(label_uint8)
            img.save(filename.replace('.png', '_gt.png'))
            
        # if flow is not None:
        #     flow_uint8 = apply_colors_to_array(x=flow, mask=None, cmap='GnBu')
        #     img = Image.fromarray(flow_uint8)
        #     img.save(filename.replace('.png', '_c.png'))
    
    def compute_average(self):
        self.avg_results = \
        {domain: {metric: value / self.num_samples 
                   for metric, value in value_dict.items()} 
            for domain, value_dict in self.sum_results.items()}
    
    def show_average_results(self):
        # Create markdown table header
        table =  "|  Domain  |  MAE  |  RMSE  |   R2   |   SSIM   |   PSNR  |\n"
        table += "|----------|-------|--------|--------|----------|---------|\n"
        
        # Add rows for each domain
        for domain, metrics in self.avg_results.items():
            table += f"| {domain} | {metrics['MAE']:.4f} | {metrics['RMSE']:.4f}| {metrics['R2']:.4f} |{metrics['SSIM']:.4f} | {metrics['PSNR']:.4f} |\n"
        
        print("\nEvaluation Results:\n" + table)
        