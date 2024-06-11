import json
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np

plt.style.use('/home/cinterno/storage/FedLEx/FedLEx-main/src/plot_style.txt')

def plot_metric_server(json_data, metric_name, save_path):
    iterations = []
    metric_values = []

    for iteration, data in json_data.items():
        if 'server_evaluated' in data and metric_name in data['server_evaluated']['metrics']:
            metric_value = data['server_evaluated']['metrics'][metric_name]
            iterations.append(iteration)
            metric_values.append(metric_value)

            plt.plot(iterations, metric_values, marker='o')
            plt.xlabel('FL round')
            plt.ylabel(metric_name)
            plt.title(f'Server evaluation for {metric_name} Metric')
            if save_path:
                plt.savefig(save_path)
                plt.clf()

def plot_metric_client(json_data, metric_name, save_path):
    iterations = []
    metric_values = []

    for iteration, data in json_data.items():
        if 'clients_updated' in data and metric_name in data['clients_updated']:
            metric_value = data['clients_updated'][metric_name]['equal']
            iterations.append(iteration)
            metric_values.append(metric_value)

            plt.plot(iterations, metric_values, marker='o')
            plt.xlabel('FL round')
            plt.ylabel(metric_name)
            plt.title(f'Clients evaluation for {metric_name} Metric')
            if save_path:
                plt.savefig(save_path)
                plt.clf()

        # Trajectory Visualization
def plot_Trajectory_l2(trajectories, save_path_t):
    iterations = list(range(0, len(trajectories)))  # Start from 1 as we need a previous model for comparison

    # Identify unique layers by removing weight/bias distinction
    unique_layers = set(['.'.join(key.split('.')[:-1]) for key in trajectories[0].keys()])

    # Initialize dictionary to store L2 diffs for each layer
    layerwise_diffs = {key: [] for key in trajectories[0].keys()}

    for i in iterations:
        # For each layer in the model
        for key in trajectories[i].keys():
            l2_diff = torch.norm(trajectories[i][key] - trajectories[i - 1][key]).item()
            layerwise_diffs[key].append(l2_diff)

    # Get global max L2-difference for consistent y-axis scaling
    max_diff = max([max(v) for v in layerwise_diffs.values()])

    # Plotting
    plt.figure(figsize=(15, len(unique_layers) * 5))

    for idx, layer in enumerate(unique_layers, 1):
        # Plot weights
        plt.subplot(len(unique_layers), 2, 2 * idx - 1)
        weight_key = f'{layer}.weight'
        plt.plot(iterations, layerwise_diffs[weight_key], label=weight_key, color='blue')
        plt.ylim(0, max_diff)
        plt.title(f'{weight_key} Trajectory (L2)')
        plt.xlabel('Training Iteration')
        plt.ylabel('L2-norm Difference')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot biases
        plt.subplot(len(unique_layers), 2, 2 * idx)
        bias_key = f'{layer}.bias'
        plt.plot(iterations, layerwise_diffs[bias_key], label=bias_key, color='red')
        plt.ylim(0, max_diff)
        plt.title(f'{bias_key} Trajectory (L2)')
        plt.xlabel('Training Iteration')
        plt.ylabel('L2-norm Difference')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if save_path_t:
        plt.savefig(f'{save_path_t}.pdf')
        plt.clf()

        data_to_save = {
            "iterations": iterations,
            "layerwise_diffs": layerwise_diffs
        }
        with open(f'{save_path_t}.json', "w") as json_file:
            json.dump(data_to_save, json_file)



def top_eigenvalues(model, data, target, criterion, k=5):
    """
    Estimate the top eigenvalue of the Hessian matrix for the given model, data, and target.

    Args:
    - model (torch.nn.Module): The neural network model.
    - data (torch.Tensor): Input data sample.
    - target (torch.Tensor): Corresponding target values.
    - criterion (torch.nn.Module): The loss function.
    - k (int): Number of power iterations to use for eigenvalue estimation.

    Returns:
    - float: Estimated top eigenvalue of the Hessian matrix.
    """

    # Ensure model gradients are zeroed
    model.zero_grad()

    # Compute the loss and its gradient with respect to model parameters
    outputs = model(data)
    loss = criterion(outputs, target)
    grads = grad(loss, model.parameters(), create_graph=True)
    grads = torch.cat([g.view(-1) for g in grads])

    # Power iteration to estimate the top eigenvalue
    v = torch.randn(grads.shape[0], device=grads.device)
    for _ in range(k):
        # Compute the Hessian-vector product
        hessian_v_prod = grad(grads, model.parameters(), grad_outputs=v, retain_graph=True)
        hessian_v_prod = torch.cat([g.contiguous().view(-1) for g in hessian_v_prod])

        # Normalize the vector to ensure numerical stability
        v = F.normalize(hessian_v_prod)

    # The top eigenvalue is given by the dot product between the vector and the Hessian-vector product
    eigenvalue = torch.dot(v, hessian_v_prod).item()

    return eigenvalue


# Helper function to collect class distributions
def get_class_distributions(split_map, dataset):
    distributions = []
    _, unique_inverse = np.unique(dataset.targets, return_inverse=True)
    for indices in split_map.values():
        client_data = np.array(dataset.targets)[indices]
        class_counts = dict(zip(*np.unique(client_data, return_counts=True)))
        distributions.append(class_counts)
    return distributions


# Plotting function
def plot_class_distributions(distributions, num_classes, save_path_distri):
    clients = list(range(len(distributions)))
    bottom = np.zeros(len(clients))

    for i in range(num_classes):
        counts = [dist.get(i, 0) for dist in distributions]
        plt.bar(clients, counts, bottom=bottom, label=f"Class {i}")
        bottom += counts

    plt.xlabel("Clients")
    plt.ylabel("Number of samples")
    plt.xlim(0, len(distributions) - 1)  # Set x-axis limits to show only the data range
    # plt.title("Class distribution per client")
    plt.legend(loc="upper right",fontsize='small')

    if save_path_distri:
        plt.savefig(f'{save_path_distri}.pdf')
        plt.clf()


