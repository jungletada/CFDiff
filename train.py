import os
import datetime
import argparse
import logging
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


from src.deepcfd.models import build_model
from data_utils.cfd_dataset import CFDDataset


def get_args():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--num_epochs', type=int, 
                        default=2000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, 
                        default=10, help='Batch size')
    parser.add_argument('--learning_rate', type=float, 
                        default=1e-4, help='Learning rate')
    parser.add_argument('--log_interval', type=int, 
                        default=20, help='Log interval')
    parser.add_argument('--root_dir', type=str, 
                        default='data/case_data1/fluent_data_map', help='Root directory for data')
    parser.add_argument('--model_type', type=str, 
                        default='unetexmod', help='Specify the type of model to use for training')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='checkpoints', help='Directory where model checkpoints will be saved')
    
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
    # loss = loss_P + loss_T + loss_V
    return loss_P, loss_T, loss_V
    

def main():
    # Initialize distributed training
    setup_ddp()
    # Set device
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    args = get_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # logger setup
    logging.basicConfig(
        filename=os.path.join(args.checkpoint_dir, f'train_{args.model_type}.log'), 
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)
    
    if rank == 0:
        logging.info("Training arguments:")
        for arg in vars(args):
            logging.info(f"  {arg}: {getattr(args, arg)}")
            
    dataset = CFDDataset(args.root_dir, is_train=True)

    sampler = DistributedSampler(
        dataset, 
        num_replicas=dist.get_world_size(), 
        rank=rank, 
        shuffle=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=dist.get_world_size() * 2, 
        pin_memory=True)
    
    # Initialize model and move to GPU
    model = build_model(key=args.model_type)
    model = model.to(device)
    logging.info(model)
    
    model = DDP(model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=True)
    
    # Loss function and optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,)
    # **Cosine Annealing Learning Rate Scheduler**
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs, 
        eta_min=1e-6)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_type)
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        sampler.set_epoch(epoch)  # Ensure shuffle works correctly in distributed mode
        
        running_loss = 0.0
        for i, data_ in enumerate(dataloader):
            inputs, targets = data_['inputs'], data_['targets']
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss_P, loss_T, loss_V = training_criterion(inputs, outputs, targets)
            loss = loss_P + loss_T + loss_V
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log only on rank 0
            if rank == 0 and (i + 1) % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                logging.info(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{i+1}/{len(dataloader)}], "
                             f"Loss_P: {loss_P:.3f}, loss_T: {loss_T:.3f}, loss_V: {loss_V:.3f}, Avg_loss: {avg_loss:.4f}")
                running_loss = 0.0

        # Adjust the learning rate using Cosine Annealing
        scheduler.step()

        # Save checkpoint only on rank 0
        if rank == 0 and (epoch + 1) % 200 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.module.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}")
    
    logging.info("Training complete.")
    
    # Clean up DDP
    cleanup()


if __name__ == '__main__':
    main()
