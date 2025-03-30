import os
import glob
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

def extract_run_info(log_dir):
    """Extract network, dataset, algorithm and topology from the run directory name."""
    run_name = os.path.basename(log_dir)
    parts = run_name.split('_')
    
    # Complete naming convention: Date_Time_ComputerName_Network_Dataset_Algorithm_Topology_Parameters
    # Example: Mar17_19-27-16_RANDY-DESKTOP777_ResNet18_CIFAR10_C-ADMM_ring_64_0.4_64
    
    # Parts indexing is 0-based, so adjust accordingly
    network = parts[3] if len(parts) > 3 else "Unknown"
    dataset = parts[4] if len(parts) > 4 else "Unknown"
    algorithm = parts[5] if len(parts) > 5 else "Unknown"  # C-ADMM or D-SGD
    topology = parts[6] if len(parts) > 6 else "Unknown"   # topology type
    
    return network, dataset, topology, algorithm

def load_tensorboard_data(log_dir):
    """Load data from TensorBoard event files."""
    event_files = glob.glob(os.path.join(log_dir, 'event*'))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None
    
    ea = EventAccumulator(event_files[0])
    ea.Reload()
    return ea

def get_variable_data(ea, variable):
    """Extract steps and values for a given variable from event accumulator."""
    if variable in ea.Tags()['scalars']:
        events = ea.Scalars(variable)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        return steps, values
    return None, None

def plot_combined_metrics(run_data, output_dir='plots'):
    """Plot combined metrics from multiple runs."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Group runs by network and dataset
    grouped_runs = {}
    for run_dir, data in run_data.items():
        network = data['network']
        dataset = data['dataset']
        key = f"{network}_{dataset}"
        
        if key not in grouped_runs:
            grouped_runs[key] = {}
        
        grouped_runs[key][run_dir] = data
    
    # For each group (network+dataset), create combined plots
    for group_key, group_data in grouped_runs.items():
        network, dataset = group_key.split('_')
        
        # Get algorithm from the first run (assuming all runs use the same algorithm)
        first_run = next(iter(group_data.values()))
        algorithm = first_run['algorithm']
        
        # Define variables to plot based on algorithm
        if "C-ADMM" in algorithm:
            variables = [
                'test loss - train loss',
                'test loss',
                'train loss',
                'test acc',
                'train acc',
                'max consensus error',
                'avg consensus error'
            ]
        else:
            variables = [
                'test loss - train loss',
                'test loss',
                'train loss',
                'test acc',
                'train acc'
            ]
        
        # Plot each variable combining data from all runs in this group
        for var in variables:
            plt.figure(figsize=(10, 6))
            
            for run_dir, data in group_data.items():
                if var in data['data']:
                    steps, values = data['data'][var]
                    plt.plot(steps, values, label=f"{data['topology']}")
            
            plt.title(f'{var} - {network} on {dataset}')
            plt.xlabel('Step')
            plt.ylabel(var)
            plt.xlim(0, 600)  # Limit x-axis to 2500 steps
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            filename = f'{var.replace(" ", "_")}_{network}_{dataset}_combined.png'
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
        
        print(f"Combined plots for {network} on {dataset} saved to {output_dir}")

def plot_individual_runs(run_data, output_dir='plots'):
    """Plot individual metrics for each run separately."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for run_dir, data in run_data.items():
        network = data['network']
        dataset = data['dataset']
        topology = data['topology']
        algorithm = data['algorithm']
        
        # Define variables to plot based on algorithm
        if "C-ADMM" in algorithm:
            variables = [
                'test loss - train loss',
                'test loss',
                'train loss',
                'test acc',
                'train acc',
                'max consensus error',
                'avg consensus error'
            ]
            n_rows, n_cols = 3, 3
        else:
            variables = [
                'test loss - train loss',
                'test loss',
                'train loss',
                'test acc',
                'train acc'
            ]
            n_rows, n_cols = 2, 3
        
        # Create subplots for this run
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
        fig.suptitle(f'Training Metrics - {network} on {dataset} ({topology} topology)', fontsize=16)
        axes = axes.ravel()
        
        # Plot each variable
        for idx, var in enumerate(variables):
            if var in data['data']:
                steps, values = data['data'][var]
                
                ax = axes[idx]
                ax.plot(steps, values, label=var)
                ax.set_title(var)
                ax.set_xlabel('Step')
                ax.set_xlim(0, 600)  # Limit x-axis to 2500 steps
                ax.grid(True)
                ax.legend()
        
        # Remove empty subplots if any
        for idx in range(len(variables), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'metrics_{network}_{dataset}_{topology}.png'))
        plt.close()
        
        # Also save individual plots for this run
        for var in variables:
            if var in data['data']:
                steps, values = data['data'][var]
                
                plt.figure(figsize=(10, 6))
                plt.plot(steps, values, label=var)
                plt.title(f'{var} - {network} on {dataset} ({topology} topology)')
                plt.xlabel('Step')
                plt.xlim(0, 600)  # Limit x-axis to 2500 steps
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{var.replace(" ", "_")}_{network}_{dataset}_{topology}.png'))
                plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot TensorBoard metrics')
    parser.add_argument('--run_dirs', type=str, nargs='+', help='Specific run directories to plot (optional)')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--combine_only', action='store_true', help='Only generate combined plots')
    args = parser.parse_args()
    
    run_dirs = []
    if args.run_dirs:
        # Use specified run directories
        for run_dir in args.run_dirs:
            if os.path.exists(run_dir):
                run_dirs.append(run_dir)
            else:
                print(f"Run directory not found: {run_dir}")
    else:
        # Use all run directories in the 'runs' folder
        run_dirs = glob.glob('runs/*')
        if not run_dirs:
            print("No run directories found in 'runs' folder")
            exit(1)
    
    if not run_dirs:
        print("No valid run directories found")
        exit(1)
    
    print(f"Processing {len(run_dirs)} run directories")
    
    # Load data from all run directories
    run_data = {}
    for run_dir in run_dirs:
        ea = load_tensorboard_data(run_dir)
        if ea:
            network, dataset, topology, algorithm = extract_run_info(run_dir)
            
            # Determine variables based on algorithm
            if "C-ADMM" in algorithm:
                variables = [
                    'test loss - train loss',
                    'test loss',
                    'train loss',
                    'test acc',
                    'train acc',
                    'max consensus error',
                    'avg consensus error'
                ]
            else:
                variables = [
                    'test loss - train loss',
                    'test loss',
                    'train loss',
                    'test acc',
                    'train acc'
                ]
            
            # Extract data for each variable
            variable_data = {}
            for var in variables:
                steps, values = get_variable_data(ea, var)
                if steps is not None:
                    variable_data[var] = (steps, values)
            
            run_data[run_dir] = {
                'network': network,
                'dataset': dataset,
                'topology': topology,
                'algorithm': algorithm,
                'data': variable_data
            }
    
    # Generate plots
    plot_combined_metrics(run_data, args.output_dir)
    
    if not args.combine_only:
        plot_individual_runs(run_data, args.output_dir) 