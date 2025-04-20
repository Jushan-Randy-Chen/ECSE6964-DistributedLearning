import datetime
import copy
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from models import get_model
from dataset import get_dataset
from gpu_work import Worker
from config import get_config

TEST_ACCURACY = 0


def work():
    config = get_config()
    print(config)
    run(**config)

def run(num_epoch, model_name, dataset_name, mode, size,
        batch_size, lr, momentum, weight_decay,
        milestones, gamma, gpu, early_stop, seed, n_swap, path, **kwargs):
    # Add ADMM specific parameters
    rho = kwargs.get('rho', 1.0)  # ADMM penalty parameter
    # use_admm = mode == "admm"  # Check if we're using C-ADMM
    use_admm = False

    topology = kwargs.get('topology')  # Get topology type, default to 'all' if not specified
    # consensus_tolerance = kwargs.get('consensus_tolerance', 1e-4)  # Get consensus tolerance
    # max_consensus_iterations = num_epoch  # Maximum iterations for ADMM consensus
    time_varying = kwargs.get('time_varying', True)  # Whether to use time-varying graphs
    topology_change_interval = kwargs.get('topology_change_interval', 50)  # How often to change topology
    available_topologies = kwargs.get('available_topologies', ['ring','all' ,'exponential'])  # Available topologies for time-varying D-SGD
    
    algorithm = "C-ADMM" if use_admm else "D-SGD"
    if time_varying and not use_admm:
        algorithm = "TV-D-SGD"  # Time-varying D-SGD
        topology = "time_varying"
    
    run_name = f"{seed}_{model_name}_{dataset_name}_{algorithm}_{topology}_{batch_size}_{lr}_{size}"
    tb = SummaryWriter(comment=run_name)
    print(f'Running {algorithm} with {topology} topology')
    temp_train_loader, temp_test_loader, input_size, classes = get_dataset(rank=0,
                                                                           dataset_name=dataset_name,
                                                                           split=None,
                                                                           batch_size=256,
                                                                           is_distribute=False,
                                                                           seed=seed,
                                                                           path=path,
                                                                           **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    P = generate_P(topology, size)  # Use topology instead of mode for generating P
    criterion = nn.CrossEntropyLoss()

    # print(temp_train_loader.__len__())
    num_step = temp_train_loader.__len__() * 64
    worker_list = []
    for rank in range(size):
        split = [1.0 / size for _ in range(size)]
        train_loader, test_loader, input_size, classes = get_dataset(rank=rank,
                                                                     dataset_name=dataset_name,
                                                                     split=split,
                                                                     batch_size=batch_size,
                                                                     seed=seed,
                                                                     path=path,
                                                                     **kwargs)
        # print(input_size,classes)
        torch.manual_seed(rank)
        model = get_model(model_name, input_size, classes)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        schedule = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        worker = Worker(rank=rank, model=model,
                        train_loader=train_loader, test_loader=test_loader,
                        optimizer=optimizer, schedule=schedule, gpu=gpu,
                        rho=rho)  # Pass rho parameter
        worker_list.append(worker)
        
        # Set up neighbors for ADMM
        if use_admm:
            neighbors = []
            if topology == "ring":
                neighbors = [(rank - 1 + size) % size, (rank + 1) % size]
            elif topology == "directed_ring":
                # In directed ring, each node only connects to the next node
                neighbors = [(rank + 1) % size]
            elif topology == "all":
                neighbors = [i for i in range(size) if i != rank]
            elif topology == "star":
                if rank == 0:
                    neighbors = list(range(1, size))
                else:
                    neighbors = [0]
            elif topology == "meshgrid":
                # Calculate grid dimensions
                i = int(np.sqrt(size))
                while size % i != 0:
                    i -= 1
                nrow, ncol = i, size // i
                # Calculate neighbors based on grid position
                row = rank // ncol
                col = rank % ncol
                neighbors = []
                # Add horizontal neighbors
                if col > 0:
                    neighbors.append(rank - 1)
                if col < ncol - 1:
                    neighbors.append(rank + 1)
                # Add vertical neighbors
                if row > 0:
                    neighbors.append(rank - ncol)
                if row < nrow - 1:
                    neighbors.append(rank + ncol)
            elif topology == "exponential":
                # In exponential topology, each node connects to nodes with indices that are powers of 2
                neighbors = []
                for i in range(size):
                    if i != rank and (i & (i - 1)) == 0:  # Check if i is a power of 2
                        neighbors.append(i)
            worker.set_neighbors(neighbors)
        # print(train_loader.__len__())
        if train_loader.__len__() < num_step:
            num_step = train_loader.__len__()

    print(f"| num_step: {num_step}")

    if use_admm:
        print("ADMM training")
        total_step = 0
        for epoch in range(1, num_epoch + 1):
            start = datetime.datetime.now()
            for worker in worker_list:
                worker.update_iter() #update worker batch for every epoch
            
            for step in range(num_step):
                total_step += 1
                # ADMM training loop
                # Step 1: Local primal update for all workers
                losses = []
                for worker in worker_list:
                    loss = worker.step_admm_primal()
                    losses.append(loss)

                # Step 2: Exchange models between workers
                for worker in worker_list:
                    neighbor_models = {}
                    for j in worker.neighbors:
                        neighbor_models[j] = worker_list[j].model.state_dict()
                    worker.update_neighbor_models(neighbor_models)

                total_step += 1
                
                # Step 3: Dual update for all workers
                for worker in worker_list:
                    worker.step_admm_dual()
                
                temp_model = copy.deepcopy(worker_list[0].model)
                for name, param in temp_model.named_parameters():
                    for worker in worker_list[1:]:
                        param.data += worker.model.state_dict()[name].data
                    param.data /= size
        
                if total_step % 50 == 0:
                    test_all(temp_model, temp_train_loader, temp_test_loader,
                            criterion, None, total_step, tb, device, n_swap=n_swap)
                if total_step == early_stop:
                    break
                    
                end = datetime.datetime.now()
                print(f"\r| Train | epoch: {epoch}|{num_epoch}, step: {step}|{num_step}, time: {(end - start).seconds}s",
                    flush=True, end="")
            if total_step == early_stop:
                break

    else:
        print("D-SGD training")
        # Original DSGD training loop with epochs and num_step
        total_step = 0
        current_topology = topology
        
        # Initialize mixing matrix - use Metropolis weights for time-varying topologies
        if time_varying:
            P = generate_metropolis_P(current_topology, size)
        else:
            P = generate_P(topology, size)
        
        for epoch in range(1, num_epoch + 1):
            start = datetime.datetime.now()
            for worker in worker_list:
                worker.update_iter()

            for step in range(num_step):
                total_step += 1
                
                # For time-varying graphs, change topology periodically
                if time_varying and total_step % topology_change_interval == 0:
                    # Deterministic cycling through topologies
                    topology_index = (total_step // topology_change_interval) % len(available_topologies)
                    # topology_index = np.random.choice(len(available_topologies))
                    current_topology = available_topologies[topology_index]
                    
                    # Use Metropolis weights for the new topology
                    P = generate_metropolis_P(current_topology, size)
                    
                    print(f"\nChanging topology to {current_topology} at step {total_step}")
                    
                    # Log the topology change
                    tb.add_text("topology_change", f"Changed to {current_topology}", total_step)

                weight_dict_list = []
            
                for worker in worker_list:
                    weight_dict_list.append(worker.model.state_dict())
                    worker.step()
                    
                for worker in worker_list:
                    for name, param in worker.model.named_parameters():
                        param.data = torch.zeros_like(param.data)
                        for i in range(size):
                            p = P[worker.rank][i]
                            param.data += weight_dict_list[i][name].data * p
                    worker.update_grad()

                temp_model = copy.deepcopy(worker_list[0].model)
                for name, param in temp_model.named_parameters():
                    for worker in worker_list[1:]:
                        param.data += worker.model.state_dict()[name].data
                    param.data /= size

                if total_step % 50 == 0:
                    test_all(temp_model, temp_train_loader, temp_test_loader,
                            criterion, None, total_step, tb, device, dataset_name, n_swap=n_swap)
                    
                    # For time-varying graphs, log the current topology
                    if time_varying:
                        tb.add_text("current_topology", current_topology, total_step)
                        
                if total_step == early_stop:
                    break
                    
                end = datetime.datetime.now()
                print(f"\r| Train | epoch: {epoch}|{num_epoch}, step: {step}|{num_step}, time: {(end - start).seconds}s",
                    flush=True, end="")
            if total_step == early_stop:
                break
    

    # checkpoint_dir = "./checkpoints/"
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    
    # # Instead of saving the entire worker objects, extract just the models
    # worker_models = []
    # for worker in worker_list:
    #     # Save only the model state dict from each worker
    #     worker_models.append(worker.model.state_dict())
    
    # # Create checkpoint with all necessary information for stability evaluation
    # checkpoint = {
    #     'worker_models': worker_models,  # Save only the model state dicts
    #     'split': [1.0 / size for _ in range(size)],
    #     'topology': topology,
    #     'algorithm': algorithm,
    #     'dataset_name': dataset_name,
    #     'model_name': model_name,
    #     'seed': seed,
    #     'total_steps': total_step
    # }
    
    # # Save checkpoint
    # checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_checkpoint.pt")
    # torch.save(checkpoint, checkpoint_path)
    # print(f"\nSaved checkpoint with worker models to: {checkpoint_path}")


def test_all(model, train_loader, test_loader, criterion, epoch, total_step, tb, device, dataset_name, n_swap=None):
    print(f"\n| Test All |", flush=True, end="")
    model.to(device)
    model.eval()
    total_loss, total_correct, total, step = 0, 0, 0, 0
    start = datetime.datetime.now()
    for batch in train_loader:
        step += 1
        data, target = batch[0].to(device), batch[1].to(device)
        output = model(data)
        p = torch.softmax(output, dim=1).argmax(1)
        total_correct += p.eq(target).sum().item()
        total += len(target)
        loss = criterion(output, target)
        total_loss += loss.item()
        end = datetime.datetime.now()
        print(f"\r| Test All |step: {step}, time: {(end - start).seconds}s", flush=True, end="")
    total_train_loss = total_loss / step
    total_train_acc = total_correct / total
    if epoch is None:
        print(f'\n| Test All Train Set |'
              f' total step: {total_step},'
              f' loss: {total_train_loss:.4},'
              f' acc: {total_train_acc:.4%}', flush=True)
    else:
        print(f'\n| Test All Train Set |'
              f' epoch: {epoch},'
              f' loss: {total_train_loss:.4},'
              f' acc: {total_train_acc:.4%}', flush=True)

    total_loss, total_correct, total, step = 0, 0, 0, 0
    for batch in test_loader:
        step += 1
        data, target = batch[0].to(device), batch[1].to(device)
        output = model(data)
        p = torch.softmax(output, dim=1).argmax(1)
        total_correct += p.eq(target).sum().item()
        total += len(target)
        loss = criterion(output, target)
        total_loss += loss.item()
        end = datetime.datetime.now()
        print(f"\r| Test All |step: {step}, time: {(end - start).seconds}s", flush=True, end="")
    total_test_loss = total_loss / step
    total_test_acc = total_correct / total
    if epoch is None:
        print(f'\n| Test All Test Set |'
              f' total step: {total_step},'
              f' loss: {total_test_loss:.4},'
              f' acc: {total_test_acc:.4%}', flush=True)
    else:
        print(f'\n| Test All Test Set |'
              f' epoch: {epoch},'
              f' loss: {total_test_loss:.4},'
              f' acc: {total_test_acc:.4%}', flush=True)

    if epoch is None:
        tb.add_scalar("test loss - train loss", total_test_loss - total_train_loss, total_step)
        tb.add_scalar("test loss", total_test_loss, total_step)
        tb.add_scalar("train loss", total_train_loss, total_step)
        tb.add_scalar("test acc", total_test_acc, total_step)
        tb.add_scalar("train acc", total_train_acc, total_step)
    else:
        tb.add_scalar("test loss - train loss", total_test_loss - total_train_loss, epoch)
        tb.add_scalar("test loss", total_test_loss, epoch)
        tb.add_scalar("train loss", total_train_loss, epoch)
        tb.add_scalar("test acc", total_test_acc, epoch)
        tb.add_scalar("train acc", total_train_acc, epoch)

    if n_swap is not None:
        if not os.path.exists("./trained/"):
            os.mkdir("./trained/")
        
        # Get algorithm and topology from the run name
        run_name = tb.log_dir.split('/')[-1]
        algorithm = "C-ADMM" if "C-ADMM" in run_name else "D-SGD"
        topology = run_name.split('_')[4]  # Get topology from run name
        
        # Create a more descriptive model name
        model_name = f"resnet18_{dataset_name}_{algorithm}_{topology}_{n_swap}"
        
        if total_test_acc > TEST_ACCURACY:
            torch.save(model.state_dict(), f"./trained/{model_name}_best.pt")
        torch.save(model.state_dict(), f"./trained/{model_name}_last.pt")


def generate_P(mode, size):
    result = torch.zeros((size, size))
    if mode == "all":
        result = torch.ones((size, size)) / size
    elif mode == "single":
        for i in range(size):
            result[i][i] = 1
    elif mode == "ring":
        for i in range(size):
            result[i][i] = 1 / 3
            result[i][(i - 1 + size) % size] = 1 / 3
            result[i][(i + 1) % size] = 1 / 3
    elif mode == "star":
        for i in range(size):
            result[i][i] = 1 - 1 / size
            result[0][i] = 1 / size
            result[i][0] = 1 / size
    elif mode == "meshgrid":
        assert size > 0
        i = int(np.sqrt(size))
        while size % i != 0:
            i -= 1
        shape = (i, size // i)
        nrow, ncol = shape
        print(shape, flush=True)
        topo = np.zeros((size, size))
        for i in range(size):
            topo[i][i] = 1.0
            if (i + 1) % ncol != 0:
                topo[i][i + 1] = 1.0
                topo[i + 1][i] = 1.0
            if i + ncol < size:
                topo[i][i + ncol] = 1.0
                topo[i + ncol][i] = 1.0
        topo_neighbor_with_self = [np.nonzero(topo[i])[0] for i in range(size)]
        for i in range(size):
            for j in topo_neighbor_with_self[i]:
                if i != j:
                    topo[i][j] = 1.0 / max(len(topo_neighbor_with_self[i]),
                                           len(topo_neighbor_with_self[j]))
            topo[i][i] = 2.0 - topo[i].sum()
        result = torch.tensor(topo, dtype=torch.float)
    elif mode == "exponential":
        x = np.array([1.0 if i & (i - 1) == 0 else 0 for i in range(size)])
        x /= x.sum()
        topo = np.empty((size, size))
        for i in range(size):
            topo[i] = np.roll(x, i)
        result = torch.tensor(topo, dtype=torch.float)
    print(result, flush=True)
    return result


def generate_metropolis_P(topology, size):
    """
    Generate a mixing matrix using Metropolis weights for the given topology.
    
    Metropolis weights formula:
    W_ij = 1/(1 + max(d_i, d_j)) if (j,i) is an edge
    W_ij = 0 if (j,i) is not an edge and j≠i
    W_ii = 1 - sum(W_il) for all l in neighbors of i
    
    Args:
        topology (str): The topology to use ('ring', 'all', 'meshgrid', 'exponential', etc.)
        size (int): Number of nodes in the network
        
    Returns:
        torch.Tensor: Size x Size mixing matrix with Metropolis weights
    """
    # Step 1: Generate adjacency matrix based on topology (without weights)
    adjacency = torch.zeros((size, size))
    
    if topology == "all":
        # Fully connected graph: all nodes connected to all others
        adjacency = torch.ones((size, size))
        # Remove self-loops for degree calculation
        for i in range(size):
            adjacency[i][i] = 0
            
    elif topology == "ring":
        # Ring topology: each node connected to left and right neighbors
        for i in range(size):
            adjacency[i][(i - 1 + size) % size] = 1  # Left neighbor
            adjacency[i][(i + 1) % size] = 1         # Right neighbor
            
    elif topology == "meshgrid":
        # Meshgrid topology: 2D grid
        i = int(np.sqrt(size))
        while size % i != 0:
            i -= 1
        shape = (i, size // i)
        nrow, ncol = shape
        
        for i in range(size):
            row, col = i // ncol, i % ncol
            # Connect to right neighbor if not at right edge
            if col < ncol - 1:
                adjacency[i][i + 1] = 1
                adjacency[i + 1][i] = 1
            # Connect to bottom neighbor if not at bottom edge
            if row < nrow - 1:
                adjacency[i][i + ncol] = 1
                adjacency[i + ncol][i] = 1
                
    elif topology == "exponential":
        # Exponential topology: connect to nodes at power-of-2 distances
        for i in range(size):
            for j in range(size):
                if i != j and (j & (j - 1)) == 0:  # j is a power of 2
                    adjacency[i][(i + j) % size] = 1
                    adjacency[(i + j) % size][i] = 1
    
    # Step 2: Calculate node degrees
    degrees = torch.sum(adjacency, dim=1)  # Sum across rows to get degrees
    
    # Step 3: Apply Metropolis weights formula
    result = torch.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            if i != j and adjacency[i][j] > 0:
                # Nodes i and j are connected
                result[i][j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
    
    # Step 4: Set diagonal entries to ensure row sum is 1 (doubly stochastic)
    for i in range(size):
        result[i][i] = 1.0 - torch.sum(result[i])
    
    # Verify that matrix is doubly stochastic (all rows and columns sum to 1)
    row_sums = torch.sum(result, dim=1)
    col_sums = torch.sum(result, dim=0)
    
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Row sums are not 1"
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5), "Column sums are not 1"
    
    print(f"Metropolis weight matrix for {topology} topology:", flush=True)
    print(result, flush=True)
    
    return result


if __name__ == '__main__':
    work()
