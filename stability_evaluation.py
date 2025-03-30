import torch
import numpy as np
import random
import copy
from torch.utils.data import Dataset, DataLoader
from dataset.distributed_dataset import DistributedDataset
from typing import Dict, List, Tuple, Any

class PerturbedDistributedDataset(DistributedDataset):
    """Extension of DistributedDataset that allows replacing a specific sample"""
    def __init__(self, dataset: Dataset, index, perturbed_idx=None, replacement_sample=None):
        super().__init__(dataset, index)
        self.perturbed_idx = perturbed_idx
        self.replacement_sample = replacement_sample
        
    def __getitem__(self, item):
        # If this is the item to be replaced and we have a replacement
        if self.perturbed_idx is not None and item == self.perturbed_idx and self.replacement_sample is not None:
            return self.replacement_sample
        return super().__getitem__(item)

def create_perturbed_datasets(original_dataset, split, size, seed=777, num_samples_to_perturb=10):
    """
    Create datasets where individual samples are replaced with new samples
    
    Args:
        original_dataset: The original dataset
        split: Distribution split across workers
        size: Number of workers
        seed: Random seed for reproducibility
        num_samples_to_perturb: Number of samples to perturb for stability evaluation
        
    Returns:
        List of (original dataset, perturbed dataset) pairs for each worker
    """
    # Create the original distributed datasets for each worker
    original_distributed_datasets = []
    for rank in range(size):
        dataset = distributed_dataset(original_dataset, split, rank, seed=seed)
        original_distributed_datasets.append(dataset)
    
    # Select random samples to perturb
    random.seed(seed)
    samples_to_perturb = random.sample(range(len(original_dataset)), num_samples_to_perturb)
    
    # For each sample to perturb, create perturbed datasets for all workers
    perturbed_datasets_list = []
    
    for sample_idx in samples_to_perturb:
        # Get a replacement sample (different from the original)
        replacement_idx = (sample_idx + 1) % len(original_dataset)
        replacement_sample = original_dataset[replacement_idx]
        
        # Create perturbed datasets for each worker
        perturbed_worker_datasets = []
        for rank, original_dataset in enumerate(original_distributed_datasets):
            # Find if this sample is in this worker's dataset
            try:
                local_idx = original_dataset.index.index(sample_idx)
                # Create perturbed dataset with replacement
                perturbed_dataset = PerturbedDistributedDataset(
                    original_dataset.dataset, 
                    original_dataset.index,
                    local_idx, 
                    replacement_sample
                )
            except ValueError:
                # Sample not in this worker's dataset, use original
                perturbed_dataset = original_dataset
                
            perturbed_worker_datasets.append(perturbed_dataset)
            
        perturbed_datasets_list.append((original_distributed_datasets, perturbed_worker_datasets))
    
    return perturbed_datasets_list

def distributed_on_average_stability(workers, dataset, split, num_samples=1, seed=777):
    """
    Evaluate distributed on-average stability by measuring how model weights change
    when individual training samples are replaced.
    
    Args:
        workers: List of worker objects
        dataset: Original dataset
        split: Distribution split across workers
        num_samples: Number of samples to perturb
        seed: Random seed for reproducibility
        
    Returns:
        stability_metric: The calculated stability metric
    """
    size = len(workers)
    
    # Create perturbed datasets
    perturbed_datasets = create_perturbed_datasets(
        dataset, split, size, seed, num_samples
    )
    
    # Save original models and loaders
    original_models = [copy.deepcopy(worker.model) for worker in workers]
    original_loaders = [worker.train_loader for worker in workers]
    
    stability_values = []
    
    # For each perturbed sample
    for i, (original_datasets, perturbed_datasets) in enumerate(perturbed_datasets):
        print(f"Evaluating stability for perturbed sample {i+1}/{num_samples}")
        
        # Create data loaders for perturbed datasets
        perturbed_loaders = [
            DataLoader(dataset, batch_size=original_loaders[j].batch_size, shuffle=True)
            for j, dataset in enumerate(perturbed_datasets)
        ]
        
        # Train on perturbed datasets
        for j, worker in enumerate(workers):
            # Set perturbed loader
            worker.train_loader = perturbed_loaders[j]
            worker.train_loader_iter = worker.train_loader.__iter__()
            
            # Reset model to original state
            worker.model = copy.deepcopy(original_models[j])
            
            # Train for a few steps
            for _ in range(200):  # Adjust number of steps as needed
                worker.step()
                worker.optimizer.step()
        
        # Calculate stability metric for this perturbation
        squared_norm_sum = 0
        for j, worker in enumerate(workers):
            # Extract last layer weights
            for name, param in worker.model.named_parameters():
                if "fc" in name or "classifier" in name:  # Assuming last layer has "fc" or "classifier" in name
                    original_param = None
                    for name_orig, param_orig in original_models[j].named_parameters():
                        if name_orig == name:
                            original_param = param_orig
                            break
                    
                    if original_param is not None:
                        # Calculate squared L2 norm of difference
                        squared_norm = torch.norm(param.data - original_param.data) ** 2
                        squared_norm_sum += squared_norm.item()
        
        stability_values.append(squared_norm_sum / size)
    
    # Restore original loaders
    for i, worker in enumerate(workers):
        worker.train_loader = original_loaders[i]
        worker.train_loader_iter = worker.train_loader.__iter__()
        worker.model = original_models[i]
    
    # Calculate final stability metric (average over all perturbations)
    stability_metric = sum(stability_values) / (num_samples * size)
    
    return stability_metric

def evaluate_stability(workers, dataset, split, num_samples=1, seed=777):
    """
    Wrapper function to evaluate distributed on-average stability
    
    Args:
        workers: List of worker objects
        dataset: Original dataset
        split: Distribution split across workers
        num_samples: Number of samples to perturb
        seed: Random seed for reproducibility
        
    Returns:
        stability_metric: The calculated stability metric
    """
    print("Evaluating distributed on-average stability...")
    stability = distributed_on_average_stability(
        workers, dataset, split, num_samples, seed
    )
    print(f"Distributed on-average stability: {stability:.6f}")
    return stability


# Add this function after the existing evaluate_stability function
def distributed_on_average_stability_time_varying(workers, dataset, split, num_samples=1, seed=777, 
                                                topology_change_interval=25, topologies=None):
    """
    Evaluate distributed on-average stability for time-varying graphs by measuring how model weights 
    change when individual training samples are replaced.
    
    Args:
        workers: List of worker objects
        dataset: Original dataset
        split: Distribution split across workers
        num_samples: Number of samples to perturb
        seed: Random seed for reproducibility
        topology_change_interval: How often to change topology (in steps)
        topologies: List of topologies to cycle through
        
    Returns:
        stability_metric: The calculated stability metric
    """
    if topologies is None:
        topologies = ['ring', 'all', 'exponential']
    
    size = len(workers)
    
    # Create perturbed datasets
    perturbed_datasets = create_perturbed_datasets(
        dataset, split, size, seed, num_samples
    )
    
    # Save original models and loaders
    original_models = [copy.deepcopy(worker.model) for worker in workers]
    original_loaders = [worker.train_loader for worker in workers]
    
    stability_values = []
    
    # For each perturbed sample
    for i, (original_datasets, perturbed_datasets) in enumerate(perturbed_datasets):
        print(f"Evaluating stability for perturbed sample {i+1}/{num_samples}")
        
        # Create data loaders for perturbed datasets
        perturbed_loaders = [
            DataLoader(dataset, batch_size=original_loaders[j].batch_size, shuffle=True)
            for j, dataset in enumerate(perturbed_datasets)
        ]
        
        # Train on perturbed datasets with time-varying graphs
        for j, worker in enumerate(workers):
            # Set perturbed loader
            worker.train_loader = perturbed_loaders[j]
            worker.train_loader_iter = worker.train_loader.__iter__()
            
            # Reset model to original state
            worker.model = copy.deepcopy(original_models[j])
            
            # Initialize topology tracking
            current_topology_idx = 0
            steps = 0
            
            # Train for a fixed number of steps
            for _ in range(100):  # Same as in original stability evaluation
                # Change topology at intervals
                if steps % topology_change_interval == 0:
                    current_topology = topologies[current_topology_idx % len(topologies)]
                    current_topology_idx += 1
                    
                    # Update worker's neighbors based on topology
                    neighbors = get_topology_neighbors(j, size, current_topology)
                    worker.set_neighbors(neighbors)
                
                # Training step with proper error handling
                try:
                    worker.step()
                    worker.optimizer.step()
                except StopIteration:
                    # Reset data loader when we reach the end
                    worker.train_loader_iter = worker.train_loader.__iter__()
                    worker.step()
                    worker.optimizer.step()
                
                steps += 1
        
        # Calculate stability metric for this perturbation
        squared_norm_sum = 0
        for j, worker in enumerate(workers):
            # Compare all model parameters
            current_params = torch.cat([p.data.view(-1) for p in worker.model.parameters()])
            original_params = torch.cat([p.data.view(-1) for p in original_models[j].parameters()])
            squared_norm = torch.norm(current_params - original_params, p=2) ** 2
            squared_norm_sum += squared_norm.item()
        
        stability_values.append(squared_norm_sum / size)
    
    # Restore original loaders
    for i, worker in enumerate(workers):
        worker.train_loader = original_loaders[i]
        worker.train_loader_iter = worker.train_loader.__iter__()
        worker.model = original_models[i]
    
    # Calculate final stability metric (average over all perturbations)
    stability_metric = sum(stability_values) / (num_samples * size)
    print(stability_metric)
    
    return stability_metric

def get_topology_neighbors(rank, size, topology):
    """Get neighbors for a worker based on topology"""
    if topology == 'ring':
        return [(rank-1) % size, (rank+1) % size]
    elif topology == 'all':
        return [i for i in range(size) if i != rank]
    elif topology == 'exponential':
        neighbors = []
        for i in range(1, int(np.log2(size)) + 1):
            neighbors.append((rank + 2**(i-1)) % size)
            neighbors.append((rank - 2**(i-1)) % size)
        return list(set([n % size for n in neighbors]))
    elif topology == 'meshgrid':
        # Implement 2D grid topology
        dim = int(np.sqrt(size))
        row, col = rank // dim, rank % dim
        neighbors = []
        if row > 0: neighbors.append(rank - dim)
        if row < dim-1: neighbors.append(rank + dim)
        if col > 0: neighbors.append(rank - 1)
        if col < dim-1: neighbors.append(rank + 1)
        return neighbors
    else:
        return []

def evaluate_stability_time_varying(workers, dataset, split, num_samples=1, seed=777, 
                                  topology_change_interval=50, topologies=None):
    """
    Wrapper function to evaluate distributed on-average stability with time-varying graphs
    
    Args:
        workers: List of worker objects
        dataset: Original dataset
        split: Distribution split across workers
        num_samples: Number of samples to perturb
        seed: Random seed for reproducibility
        topology_change_interval: How often to change topology (in steps)
        topologies: List of topologies to cycle through
        
    Returns:
        stability_metric: The calculated stability metric
    """
    print("Evaluating distributed on-average stability with time-varying graphs...")
    stability = distributed_on_average_stability_time_varying(
        workers, dataset, split, num_samples, seed, topology_change_interval, topologies
    )
    print(f"Distributed on-average stability (time-varying): {stability:.6f}")
    return stability




# Helper function to get the distributed dataset for a specific worker
def distributed_dataset(dataset: Dataset, split: Any, rank: int, size: int = None, seed: int = 777):
    if size is None:
        size = len(dataset)
    random.seed(seed)
    indexes = [x for x in range(size)]
    random.shuffle(indexes)
    indexes_list = []
    for s in split:
        indexes_list.append(indexes[:int(s * size)])
        indexes = indexes[int(s * size):]
    return DistributedDataset(dataset, indexes_list[rank])
