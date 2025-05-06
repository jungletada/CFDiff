import os
import argparse
import logging
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


from src.deepcfd.models import get_model
from src.deepcfd.models.UNetEx import UNetEx
from data_utils.cfd_dataset import CFDDataset


def get_args():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--num_epochs', type=int, default=3000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=20, help='Log interval')
    parser.add_argument('--root_dir', type=str, default='data/case_data1/fluent_data_map', help='Root directory for data')
    parser.add_argument('--model_type', type=str, default='unet', help='Specify the type of model to use for training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory where model checkpoints will be saved')
    
    args = parser.parse_args()
    return args

def setup_ddp():
    """ Initialize the distributed process group. """
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def cleanup():
    """ Clean up the process group. """
    dist.destroy_process_group()


def training_criterion(inputs, outputs, targets):
    masks = (inputs[:, 0].detach() > 0.5).float()  # 1 for field, 0 for contour
    
    mse_criterion = nn.MSELoss(reduction='mean')
    l1_criterion = nn.L1Loss(reduction='mean')
    
    loss_P = l1_criterion(outputs[:,0,:,:] * masks, targets[:,0,:,:] * masks)
    loss_T = mse_criterion(outputs[:,1,:,:] * masks, targets[:,1,:,:] * masks)
    loss_V = mse_criterion(outputs[:,2,:,:] * masks, targets[:,2,:,:] * masks)
    loss = loss_P + loss_T + loss_V
    return loss
    

def main():
    # Initialize distributed training
    setup_ddp()
    # Set device
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    
    args = get_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    log_interval = args.log_interval
    root_dir = args.root_dir
    
    dataset = CFDDataset(root_dir, mode='train')
    
    sampler = DistributedSampler(
        dataset, 
        num_replicas=dist.get_world_size(), 
        rank=rank, 
        shuffle=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=dist.get_world_size() * 2, 
        pin_memory=True)
    
    # Initialize model and move to GPU
    # model = UNetEx(in_channels=2, out_channels=3).to(device)
    model_type = get_model(key=args.model_type)
    
    model = model_type(in_channels=2, 
                       out_channels=3).to(device)
    model = DDP(model, 
                device_ids=[local_rank], 
                output_device=local_rank)
    
    # Loss function and optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate)
    # **Cosine Annealing Learning Rate Scheduler**
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs, 
        eta_min=1e-5)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_type)
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)  # Ensure shuffle works correctly in distributed mode
        
        running_loss = 0.0
        for i, data_ in enumerate(dataloader):
            inputs, targets = data_['inputs'], data_['targets']
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = training_criterion(inputs, outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log only on rank 0
            if rank == 0 and (i + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # Adjust the learning rate using Cosine Annealing
        scheduler.step()

        # Save checkpoint only on rank 0
        if rank == 0 and (epoch + 1) % 200 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.module.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
    
    print("Training complete.")
    
    # Clean up DDP
    cleanup()


if __name__ == '__main__':
    main()
