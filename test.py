import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader



from src.deepcfd.models import build_model
from eval_metrics import Evaluator
from data_utils.cfd_dataset import CFDDataset
from data_utils.cfd_dataset import \
        (STAT_pressure, STAT_temperature, STAT_velocity)
from src.deepcfd.metrics import abs_relative_difference, \
        squared_relative_difference, delta1_acc
    

def get_args():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=1, 
                        help='Batch size')
    parser.add_argument('--root_dir', 
                        type=str, 
                        default='data/case_data2/fluent_data_map', 
                        help='Root directory for data')
    parser.add_argument('--model_type', 
                        type=str, 
                        default='unet', 
                        help='Specify the type of model to use for training')
    parser.add_argument('--checkpoint_path', 
                        type=str, 
                        default='checkpoints/unet/epoch_3000.pth', 
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--num_workers', 
                        type=int, 
                        default=2, 
                        help='Number of workers for DataLoader')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # Build the test dataset and dataloader
    test_dataset = CFDDataset(
        args.root_dir, 
        is_train='test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers)
    
    # Set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model 
    # (input: contour image with 2 channel, 
    # output: 3 channels for pressure, temperature, velocity)
    model = build_model(key=args.model_type)
    model = model.to(device)
    
    # Optionally load a trained checkpoint if available
    if os.path.exists(args.checkpoint_path):
        print(f"Loading model checkpoint from {args.checkpoint_path} ...")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    else:
        print("No checkpoint found; evaluating an untrained model.")

    results_path = os.path.join("results", args.model_type)
    os.makedirs(results_path, exist_ok=True)
    # Evaluate the model on the test set using MSE
    model.eval()

    num_inputs = len(test_dataset)
    evaluator = Evaluator(num_inputs)
    
    with torch.no_grad():
        for data_dict in tqdm(test_loader):
            # Move input to device
            inputs = data_dict['inputs'].to(device)  # (B, 1, H, W)
            # Move each target modality to device and then concatenate along channel dim.
            targets = data_dict['targets'].squeeze().to(device)  # (3, H, W)
            targets = (targets + 1.) / 2. 
            targets = targets.cpu().numpy()
            # Forward pass
            outputs = model(inputs).squeeze().clip(-1.0, 1.0)
            outputs = (outputs + 1.) / 2. 
            outputs = outputs.cpu().numpy()
            # make mask
            mask = (inputs.squeeze()[0].detach() > 0.5).bool()
            mask = mask.cpu().numpy()
            
            flow = inputs.squeeze()[0].detach()
            flow[flow > 0.5] = 1
            flow[flow != 1] = 0
            flow = flow.float().cpu().numpy()
            # contour = mask.float()
            base_path = f"{results_path}/{data_dict['filepath'][0]}"
            
            evaluator.evaluate_single(outputs[0],targets[0], field='pressure', filename=base_path+"-pressure.png", mask=mask)
            evaluator.evaluate_single(outputs[1],targets[1], field='temperature', filename=base_path+"-temperature.png", mask=mask)
            evaluator.evaluate_single(outputs[2],targets[2], field='velocity', filename=base_path+"-velocity.png", mask=mask)
            
            # evaluator.visualize_single(pred=outputs[0], filename=base_path+"-pressure.png", mask=mask, field='pressure',)
            # evaluator.visualize_single(pred=outputs[1], filename=base_path+"-temperature.png", mask=mask, field='temperature',)
            # evaluator.visualize_single(pred=outputs[2], filename=base_path+"-velocity.png", mask=mask, field='velocity',)
        
    evaluator.compute_average()
    evaluator.show_average_results()


if __name__ == '__main__':
    main()
