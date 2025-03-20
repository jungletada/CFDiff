import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb

# Import model and dataset
from UNetEx import UNetEx
from dataset import CaseDataDataset


def setup_ddp():
    """ Initialize the distributed process group. """
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def cleanup():
    """ Clean up the process group. """
    dist.destroy_process_group()


def main():
    # Initialize distributed training
    setup_ddp()
    
    # Set device
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    
    # Hyperparameters
    num_epochs = 100
    batch_size = 4
    learning_rate = 1e-4
    log_interval = 10  # Log every X steps
    
    # Dataset and DataLoader (using DistributedSampler)
    root_dir = 'data/case_data1/fluent_data_fig'
    dataset = CaseDataDataset(root_dir)
    
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    
    # Initialize model and move to GPU
    model = UNetEx(in_channels=1, out_channels=3).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Only rank 0 initializes wandb
    if rank == 0:
        wandb.init(
            entity="dingjie-peng-waseda-university",
            project="small-demo",
            config={
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "architecture": "UNetEx",
                "distributed": True}
            )

    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)  # Ensure shuffle works correctly in distributed mode
        
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)

            target_pressure = targets['pressure'].to(device)
            target_temperature = targets['temperature'].to(device)
            target_velocity = targets['velocity'].to(device)
            
            targets_tensor = torch.cat([target_pressure, target_temperature, target_velocity], dim=1)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Log only on rank 0
            if rank == 0 and (i + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {avg_loss:.4f}")
                wandb.log({"Loss": avg_loss})
                running_loss = 0.0
        
        # Save checkpoint only on rank 0
        if rank == 0 and (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.module.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
    
    print("Training complete.")
    
    # Clean up DDP
    cleanup()

if __name__ == '__main__':
    main()
