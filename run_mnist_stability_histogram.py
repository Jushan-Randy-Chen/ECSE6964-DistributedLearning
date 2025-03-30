import torch
import importlib
import os
import numpy as np
from config import get_config
from stability_evaluation import evaluate_stability, evaluate_stability_time_varying
from gpu_work import Worker

def main():
    # Get configuration
    config = get_config()
    
    # Override config for MNIST with ResNet18
    config['dataset_name'] = 'MNIST'
    config['model_name'] = 'ResNet18'
    config['batch_size'] = 64
    config['stability_num_samples'] = 1
    
    # Time-varying graph parameters
    topology_change_interval = 50
    topologies = ['ring', 'all', 'exponential']
    
    torch.manual_seed(config['seed'])
    
    # Import dataset module
    dataset_module = importlib.import_module(f"dataset.{config['dataset_name'].lower()}")
    dataset_func = getattr(dataset_module, config['dataset_name'].lower())
    
    # Import model module
    model_module = importlib.import_module("models.resnet")
    model_func = getattr(model_module, config['model_name'])
    
    # Load dataset FIRST
    train_loader, test_loader, _, _ = dataset_func(
        rank=0,
        split=[1.0],
        batch_size=config['batch_size'],
        is_distribute=False,
        path=config['path']
    )
    original_dataset = train_loader.dataset
    
    # Create workers with actual data loaders
    print("Creating fresh models for stability evaluation")
    workers = []
    for i in range(16):
        model = model_func(in_channel=1, classes=10)
        # Create optimizer with proper learning rate
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        worker = Worker(
            rank=i,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            schedule=None,
            gpu=False,
        )
        workers.append(worker)
    print(f"Created {len(workers)} workers with valid data loaders")
    
    # Evaluate stability with time-varying graphs
    split = [1.0/16] * 16
    stability = evaluate_stability_time_varying(
        workers,
        original_dataset,
        split,
        num_samples=config['stability_num_samples'],
        seed=config['seed'],
        topology_change_interval=topology_change_interval,
        topologies=topologies
    )
    
    # Save results
    results_dir = "stability_results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "mnist_resnet18_time_varying_stability.txt")
    
    with open(results_file, 'w') as f:
        f.write(f"Number of workers: {len(workers)}\n")
        f.write(f"Topologies: {', '.join(topologies)}\n")
        f.write(f"Topology change interval: {topology_change_interval}\n")
        f.write(f"Distributed on-average stability: {stability:.6f}\n")
    
    print(f"Results saved to {results_file}")
    print(f"Stability value: {stability:.6f}")

if __name__ == "__main__":
    main()