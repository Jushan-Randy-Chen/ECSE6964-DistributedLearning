from typing import List, Dict

from torch.utils.data import DataLoader
from torch.nn import Module
import torch.nn as nn

import torch


class Worker:
    def __init__(self, rank, model: Module,
                 train_loader: DataLoader, test_loader: DataLoader,
                 optimizer, schedule,
                 gpu=True, rho=1.0):
        self.rank = rank
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_loader_iter = train_loader.__iter__()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.schedule = schedule
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        self.model.to(self.device)
        
        # ADMM variables
        self.rho = rho  # ADMM penalty parameter
        self.lambda_dict = {}  # Dual variables
        self.neighbors = []  # List of neighbor worker ranks
        self.communication_weights = {}  # Communication weights for each neighbor
        self.neighbor_models = {}  # Dictionary to store neighbor models
        
        # Initialize ADMM variables
        """To see why we initialize the dual variable p_i^{kT} like this, example: 
        {
                'layer1.weight': torch.zeros(64, 64),  # Dual variable for weight matrix
                'layer1.bias': torch.zeros(64)         # Dual variable for bias vector
                ......
            }
        """
        for name, param in self.model.named_parameters():
            self.lambda_dict[name] = torch.zeros_like(param.data)

    def update_iter(self):
        self.train_loader_iter = self.train_loader.__iter__()

    def set_neighbors(self, neighbors, P=None):
        """Set the list of neighbor workers and their communication weights"""
        self.neighbors = neighbors
        if P is not None:
            self.communication_weights = {j: P[self.rank][j] for j in neighbors}

    def step_admm_primal(self):
        """Primal update using previous dual variables and neighbor parameters."""
        self.model.train()
        data, target = self.train_loader_iter.__next__()
        data, target = data.to(self.device), target.to(self.device)
        
        # Forward pass and loss computation
        output = self.model(data)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        """C-ADMM reference: https://ieeexplore.ieee.org/document/9147590"""
        # Add gradients from dual variables and consensus terms
        for name, param in self.model.named_parameters():
            # Add dual variable gradient (λ_i^k)
            if name in self.lambda_dict:
                param.grad += self.lambda_dict[name].to(self.device)
            
            # Add consensus term gradient (using x_j^k)
            for j in self.neighbors:
                neighbor_model = self.neighbor_models.get(j, None)
                if neighbor_model and name in neighbor_model:
                    avg_state = (param.data + neighbor_model[name].to(self.device)) / 2
                    param.grad += self.rho * (param.data - avg_state)
        
        # Update parameters (x_i^{k+1})
        self.optimizer.step()
        return loss.item()

    def step_admm_dual(self):
        """Dual update using latest neighbor parameters (x_j^{k+1})."""
        for name, param in self.model.named_parameters():
            for j in self.neighbors:
                if j in self.communication_weights and name in self.neighbor_models[j]:
                    neighbor_param = self.neighbor_models[j][name].to(self.device)
                    # Update dual variable: λ_i^{k+1} = λ_i^k + ρ(x_i^{k+1} - x_j^{k+1})
                    self.lambda_dict[name] += self.rho * (param.data - neighbor_param)

    def update_neighbor_models(self, neighbor_models):
        """Update the stored neighbor models for the next ADMM step"""
        self.neighbor_models = neighbor_models

    def step(self):
        self.model.train()
        self.model.to(self.device)

        batch = self.train_loader_iter.__next__()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()

    def step_csgd(self):
        self.model.train()
        self.model.to(self.device)

        batch = self.train_loader_iter.__next__()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        output = self.model(data)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()

        grad_dict = {}
        for name, param in self.model.named_parameters():
            grad_dict[name] = param.grad.data

        return grad_dict

    def update_grad(self):
        self.optimizer.step()
        self.schedule.step()

    def schedule_step(self):
        self.schedule.step()

    def check_consensus_error(self):
        """Calculate the maximum consensus error between this worker and its neighbors"""
        max_error = 0.0
        for name, param in self.model.named_parameters():
            for j in self.neighbors:
                if j in self.communication_weights:
                    neighbor_model = self.neighbor_models.get(j, None)
                    if neighbor_model is not None and name in neighbor_model:
                        # Calculate L2 norm of parameter difference
                        error = torch.norm(param.data - neighbor_model[name])
                        max_error = max(max_error, error)
        return max_error
