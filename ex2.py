

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sympy import false
from torch.onnx.symbolic_opset9 import tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D

TRAIN_NORMALLIZE = False
TRAIN_UNCONDITONAL = False
TRAIN_CONDITIONAL = False


def plot_validation_loss(training_losses, validation_losses, log_determinant=None, log_pz=None, device='cpu'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # First subplot for training and validation losses
    ax1.plot(training_losses, label='Training Loss')
    ax1.plot(validation_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    if log_determinant and log_pz:
        # Second subplot for log determinant and log_pz
        ax2.plot(log_determinant, label='Log Determinant')
        ax2.plot(log_pz, label='Log Pz(x)')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Log Values')
        ax2.set_title('Flow Components')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig('loss_plot.png')  # Save the plot to disk
    plt.show()


def visualize_model(model_num=20, num_points=1000, seeds=[42, 108, 555], device='cpu'):
    """Visualize samples from the trained flow model using different seeds."""

    model = flow_model()
    model.load_state_dict(torch.load(f"normalize_flow_model_epoch_{model_num}.pth", map_location=device))
    model.to(device)
    model.eval()

    base_dist = MultivariateNormal(
        torch.zeros(model.input_dim, device=device),
        torch.eye(model.input_dim, device=device)
    )

    num_seeds = len(seeds)
    cols = min(num_seeds, 3)
    rows = (num_seeds + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 5 * rows))

    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)

        z = base_dist.sample((num_points,))
        x, _ = model(z)  # Assuming model returns (x, log_det) or similar
        x = x.detach().numpy()
        plt.subplot(rows, cols, i + 1)
        plt.scatter(x[:, 0], x[:, 1], s=2, alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'Seed = {seed}')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Generated Points from Normalizing Flow', fontsize=16, y=1.02)
    plt.show()
    return x
#end of normalizing flow

def sample_from_matching_flow(model, num_points=10, device='cpu', dt=0.0001, store_trajectory=False,
                              times=[0, 0.2, 0.4, 0.6, 0.8, 1]):
    model.eval()
    trajectories = []

    with torch.no_grad():
        base_dist = MultivariateNormal(torch.zeros(model.input_dim).to(device), torch.eye(model.input_dim).to(device))
        y = base_dist.sample((num_points,)).to(device)

        if store_trajectory:
            trajectories.append(y.cpu().numpy().copy())

        steps = int(1 / dt)

        for i in range(steps):
            t_scalar = dt * i
            t_tensor = torch.full((num_points, 1), t_scalar, device=device)


            v_t = model.forward(y, t_tensor)


            y = y + dt * v_t


            if store_trajectory:

                for target_time in times:
                    if abs(t_scalar - target_time) < dt / 2:
                        trajectories.append(y.cpu().numpy().copy())
                        break

    if store_trajectory:

        if 1.0 in times and len(trajectories) <= len(times):
            trajectories.append(y.cpu().numpy().copy())
        return y, trajectories

    return y

def resample_from_matching_flow(y,model, num_points=2000, device='cpu',dt=0.0001):
    model.eval()
    with torch.no_grad():
        base_dist = MultivariateNormal(torch.zeros(model.input_dim).to(device), torch.eye(model.input_dim).to(device))
        for t in range(1,0,-dt):
            y = y - dt * model.forward(y)
    return y
def visualize_matching_flow_samples(y, title='Samples from Matching Flow'):
    # Convert to NumPy if still torch.Tensor
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    # Check for NaNs or degenerate shape
    if np.isnan(y).any():
        print("❌ NaNs found in samples! Cannot visualize.")
        return
    if y.shape[1] != 2:
        raise ValueError(f"Expected 2D data for visualization, got shape {y.shape}")

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y[:, 0], y[:, 1], s=1, alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage

# region: Conditional Helpers

def generate_points_on_ring(center, radius, thickness, num_points):
    points = []
    while len(points) < num_points:
        r = np.random.uniform(radius - thickness / 2, radius + thickness / 2)
        theta = np.random.uniform(0, 2 * np.pi)
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        points.append((x, y))
    return points


def sample_olympic_rings(num_points_per_ring, ring_thickness=0.1):
    centers = [(0, 0), (2, 0), (4, 0), (1, -1), (3, -1)]
    colors = ['blue', 'black', 'red', 'yellow', 'green']
    radius = 1
    all_points = []
    all_labels = []

    for center, color in zip(centers, colors):
        points = generate_points_on_ring(center, radius, ring_thickness, num_points_per_ring)
        labels = [color] * num_points_per_ring
        all_points.extend(points)
        all_labels.extend(labels)

    return all_points, all_labels


# endregion: Conditional Helpers

# region: Unconditional Helpers

def point_in_ring(x, y, center, radius, thickness):
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return radius - thickness / 2 <= distance <= radius + thickness / 2


def generate_points_on_rings__unconditional(centers, radius, thickness, num_points):
    points = []
    count = 0
    while count < num_points:
        x = np.random.uniform(-1, 5)
        y = np.random.uniform(-2, 1)
        in_any_ring = False
        for center in centers:
            if point_in_ring(x, y, center, radius, thickness):
                in_any_ring = True
                break
        if in_any_ring:
            points.append((x, y))
            count += 1
    return points


# endregion: Unconditional Helpers


def create_olympic_rings(n_points, ring_thickness=0.25, verbose=True):
    num_points_per_ring = n_points // 5
    sampled_points, labels = sample_olympic_rings(num_points_per_ring, ring_thickness)

    # Plotting the points
    if verbose:
        x, y = zip(*sampled_points)
        colors = labels
        if len(sampled_points) > 10000:
            rand_idx = np.random.choice(len(sampled_points), 10000, replace=False)
            plt.scatter(np.array(x)[rand_idx], np.array(y)[rand_idx], s=1, c=np.array(colors)[rand_idx])
        else:
            plt.scatter(x, y, s=1, c=colors)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Numpy Sampled Olympic Rings')
        plt.show()

    sampled_points = np.asarray(sampled_points)
    # transform labels from strings to ints
    label_to_int = {k: v for v, k in enumerate(np.unique(labels))}
    int_to_label = {v: k for k, v in label_to_int.items()}
    labels = np.array([label_to_int[label] for label in labels])

    sampled_points = np.asarray(sampled_points)
    # normalize data
    sampled_points = (sampled_points - np.mean(sampled_points, axis=0)) / np.std(sampled_points, axis=0)

    return sampled_points, labels, int_to_label


def create_unconditional_olympic_rings(n_points, ring_thickness=0.25, verbose=True):
    centers = [(0, 0), (2, 0), (4, 0), (1, -1), (3, -1)]
    radius = 1
    data = generate_points_on_rings__unconditional(centers, radius, ring_thickness, n_points)
    if verbose:
        x, y = zip(*data)
        if len(data) > 10000:
            rand_idx = np.random.choice(len(data), 10000, replace=False)
            plt.scatter(np.array(x)[rand_idx], np.array(y)[rand_idx], s=1)
        else:
            plt.scatter(x, y, s=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Numpy Sampled Olympic Rings')
        plt.show()
    data = np.asarray(data)
    # normalize data
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return data

class Coupling_layer(nn.Module):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Network for scale (s) - we'll learn log(s) for numerical stability
        self.nn_s = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim // 2)
        )

        # Network for translation (b)
        self.nn_b = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim // 2)
        )

    def forward(self, z):
        z_l = z[:, :self.input_dim // 2]
        z_r = z[:, self.input_dim // 2:]

        # Get log(s) from network and clamp for stability
        log_s = self.nn_s(z_l)
        log_s = torch.clamp(log_s, min=-5.0, max=5.0)

        # Get translation
        b = self.nn_b(z_l)

        # Apply affine transformation: y = s * z + b
        # where s = exp(log_s)
        s = torch.exp(log_s)
        y_r = s * z_r + b
        y_l = z_l

        y = torch.cat((y_l, y_r), dim=1)
        return y, log_s  # Return log_s for Jacobian calculation

    def inverse(self, y):
        y_l = y[:, :self.input_dim // 2]
        y_r = y[:, self.input_dim // 2:]

        z_l = y_l

        # Get log(s) from network and clamp for stability
        log_s = self.nn_s(z_l)
        log_s = torch.clamp(log_s, min=-5.0, max=5.0)

        # Get translation
        b = self.nn_b(z_l)

        # Inverse transformation: z = (y - b) / s
        # where s = exp(log_s)
        s = torch.exp(log_s)
        z_r = (y_r - b) / s

        z = torch.cat((z_l, z_r), dim=1)
        return z, log_s  # Return log_s for Jacobian calculation

    def log_jacobian(self, log_s):
        """
        For affine coupling layer: y = s * z + b
        The Jacobian determinant is the product of scale factors s
        So log|det(J)| = sum(log(s)) = sum(log_s)
        """
        if log_s is not None:
            return log_s.sum(dim=1)  # Sum over feature dimension
        return torch.zeros(1)


class permutation_layer(nn.Module):  # Fixed typo in class name
    def __init__(self, input_dim=2):
        super().__init__()
        self.input_dim = input_dim
        # Register as buffer so it moves with the model to GPU
        self.register_buffer('permutation', torch.randperm(self.input_dim))

    def forward(self, z):
        return z[:, self.permutation], None

    def inverse(self, y):
        inverse_permutation = torch.argsort(self.permutation)
        return y[:, inverse_permutation], None

    def log_jacobian(self, log_s=None):
        # Permutation has determinant ±1, so log|det| = 0
        return torch.zeros(1, device=self.permutation.device)


class flow_model(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, layers_num=15,device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        self.layers_num = layers_num
        self.base_dist = MultivariateNormal(
                torch.zeros(self.input_dim).to(device),
                torch.eye(self.input_dim).to(device)
            )

        for _ in range(layers_num):
            self.layers.append(permutation_layer(input_dim))
            self.layers.append(Coupling_layer(input_dim, input_dim, hidden_dim))

    def forward(self, z):
        """Transform from base distribution to data distribution"""
        log_jacobian = torch.zeros(z.shape[0], device=z.device)

        for layer in self.layers:
            z, log_s = layer.forward(z)
            log_jac_contribution = layer.log_jacobian(log_s)
            if log_jac_contribution.numel() > 1:  # Not a scalar
                log_jacobian += log_jac_contribution
            # If it's a scalar (like from permutation), it's 0 anyway

        return z, log_jacobian

    def inverse(self, y):
        """Transform from data distribution to base distribution"""
        log_jacobian = torch.zeros(y.shape[0], device=y.device)

        for layer in reversed(self.layers):
            y, log_s = layer.inverse(y)
            log_jac_contribution = layer.log_jacobian(log_s)
            if log_jac_contribution.numel() > 1:  # Not a scalar
                # For inverse transformation, we subtract the log Jacobian
                log_jacobian -= log_jac_contribution

        return y, log_jacobian
    def sample_over_time(self, num_points=1000, num_figs=5, device='cpu'):
        """Visualize how samples evolve over the layers of the flow model."""
        self.eval()
        with torch.no_grad():
            # Set up the base distribution
            points = self.base_dist.sample((num_points,)).to(device)

            # Determine how often to snapshot the layers
            total_layers = len(self.layers)
            snapshot_every = max(1, total_layers // num_figs)

            # Set up subplots
            fig, axs = plt.subplots(1, num_figs, figsize=(5 * num_figs, 5))
            axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]  # handle num_figs == 1

            fig_count = 0
            for layer_idx, layer in enumerate(self.layers):
                points,_ = layer(points)
                if (layer_idx + 1) % snapshot_every == 0 or (layer_idx + 1) == total_layers:
                    ax = axs[fig_count]
                    x_np = points.detach().cpu().numpy()
                    ax.scatter(x_np[:, 0], x_np[:, 1], s=2, alpha=0.5)
                    ax.set_title(f'After Layer {layer_idx + 1}')
                    ax.set_aspect('equal', adjustable='box')
                    ax.grid(True, alpha=0.3)
                    fig_count += 1
                    if fig_count >= num_figs:
                        break

            plt.tight_layout()
            plt.suptitle("Transformation Over Layers", fontsize=16, y=1.05)
            plt.show()
    def sample_trajectories(self, num_points=10, device='cpu'):
        self.eval()
        with torch.no_grad():
            # Sample initial points
            points = self.base_dist.sample((num_points,)).to(device)
            colors = cm.viridis(torch.linspace(0, 1, len(self.layers)//2 + 1).numpy())
            plt.figure(figsize=(6, 6))
            color_step = 0  # For assigning different colors to each plotted layer

            for layer_idx, layer in enumerate(self.layers):
                points,_ = layer(points)

                # Plot every 2 layers and the final one
                if layer_idx % 2 == 0 or layer_idx + 1 == len(self.layers):
                    pts = points.detach().cpu().numpy()
                    plt.scatter(pts[:, 0], pts[:, 1], s=40, alpha=0.7, color=colors[color_step], label=f'Layer {(layer_idx + 1)//2}')
                    color_step += 1

            plt.legend()
            plt.title("Point Trajectories Through Flow Layers")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True, alpha=0.3)
            plt.show()

    def visualize_point_trajectories(self, points_on_rings, points_out_of_rings, device='cpu'):
        self.eval()


        max_traj_points = min(5, points_on_rings.shape[0])
        points_on_subset = points_on_rings[:max_traj_points]

        max_traj_points_off = min(5, points_out_of_rings.shape[0])
        points_off_subset = points_out_of_rings[:max_traj_points_off]

        # Combine points for tracking
        all_points = torch.cat([points_on_subset, points_off_subset], dim=0)
        is_on_ring = torch.cat([
            torch.ones(points_on_subset.shape[0], dtype=torch.bool),
            torch.zeros(points_off_subset.shape[0], dtype=torch.bool)
        ])

        # Track points through each layer
        point_history = [all_points.detach().cpu().numpy()]
        current_points = all_points.clone()

        for layer in self.layers:
            current_points, _ = layer.forward(current_points)
            point_history.append(current_points.detach().cpu().numpy())

        # Plot trajectories
        plt.figure(figsize=(8, 8))

        for i in range(all_points.shape[0]):
            # Extract trajectory for this point
            traj = np.array([states[i] for states in point_history])

            # Use different color based on whether point is on/off ring
            color = 'blue' if is_on_ring[i] else 'red'
            label = "On ring" if is_on_ring[i] and i == 0 else (
                "Off ring" if not is_on_ring[i] and i == points_on_subset.shape[0] else None)

            # Plot trajectory line
            plt.plot(traj[:, 0], traj[:, 1], '-', color=color, alpha=0.7, linewidth=1, label=label)

            # Mark start and end points
            plt.scatter(traj[0, 0], traj[0, 1], color=color, s=50, marker='o')
            plt.scatter(traj[-1, 0], traj[-1, 1], color=color, s=50, marker='x')

        plt.title("Point Trajectories Through Flow Layers")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def sample_inverse(self, points_on_rings, points_out_of_rings, num_snapshots=5, device='cpu'):

        self.eval()

        # Convert inputs to tensors if they aren't already
        if isinstance(points_on_rings, np.ndarray):
            points_on_rings = torch.tensor(points_on_rings, dtype=torch.float32)
        if isinstance(points_out_of_rings, np.ndarray):
            points_out_of_rings = torch.tensor(points_out_of_rings, dtype=torch.float32)

        # Move to device
        points_on_rings = points_on_rings.to(device)
        points_out_of_rings = points_out_of_rings.to(device)

        # Create a combined tensor but keep track of which points are which
        all_points = torch.cat([points_on_rings, points_out_of_rings], dim=0)
        on_ring_mask = torch.zeros(all_points.shape[0], dtype=torch.bool)
        on_ring_mask[:points_on_rings.shape[0]] = True

        # Setup for visualization
        total_layers = len(self.layers)
        snapshot_interval = max(1, total_layers // num_snapshots)
        fig, axes = plt.subplots(1, num_snapshots, figsize=(4 * num_snapshots, 4))
        if num_snapshots == 1:
            axes = [axes]

        # Track points through each layer
        current_points = all_points.clone()
        snapshot_count = 0
        log_det_sum = torch.zeros(current_points.shape[0], device=device)

        for i, layer in enumerate(self.layers):
            current_points, log_s = layer.forward(current_points)

            # Add to log determinant if available
            if log_s is not None:
                log_jac_contribution = layer.log_jacobian(log_s)
                if log_jac_contribution.numel() > 1:
                    log_det_sum += log_jac_contribution

            if (i + 1) % snapshot_interval == 0 or i == total_layers - 1:
                if snapshot_count < len(axes):
                    ax = axes[snapshot_count]
                    pts_np = current_points.detach().cpu().numpy()

                    # Plot with different colors for on-ring vs off-ring points
                    on_ring_pts = pts_np[on_ring_mask]
                    off_ring_pts = pts_np[~on_ring_mask]

                    ax.scatter(on_ring_pts[:, 0], on_ring_pts[:, 1], c='blue',
                               s=10, alpha=0.7, label='On rings' if i == 0 else None)
                    ax.scatter(off_ring_pts[:, 0], off_ring_pts[:, 1], c='red',
                               s=10, alpha=0.7, label='Off rings' if i == 0 else None)

                    ax.set_title(f"Layer {i + 1}/{total_layers}")
                    ax.set_aspect('equal', adjustable='box')
                    ax.grid(True, alpha=0.3)

                    if i == 0:
                        ax.legend()

                    snapshot_count += 1

        # Calculate log probability in base distribution
        log_prob_base = self.base_dist.log_prob(current_points)

        # Total log probability = log_prob_base + log_det_jacobian
        log_prob_x = log_prob_base + log_det_sum

        # Print log probabilities
        print("\nLog probability of points [log p(x)]:")
        print("Points on rings:")
        for i, prob in enumerate(log_prob_x[on_ring_mask]):
            print(f"  Point {i + 1}: {prob.item():.4f}")

        print("\nPoints off rings:")
        for i, prob in enumerate(log_prob_x[~on_ring_mask]):
            print(f"  Point {i + 1}: {prob.item():.4f}")

        # Calculate and print mean log probabilities for each group
        mean_prob_on_rings = log_prob_x[on_ring_mask].mean().item()
        mean_prob_off_rings = log_prob_x[~on_ring_mask].mean().item()
        print(f"\nMean log probability for points on rings: {mean_prob_on_rings:.4f}")
        print(f"Mean log probability for points off rings: {mean_prob_off_rings:.4f}")

        plt.suptitle("Transformation Through Flow Layers", fontsize=16, y=1.05)
        plt.tight_layout()
        plt.show()

        # Also visualize trajectories of individual points
        self.visualize_point_trajectories(points_on_rings, points_out_of_rings, device)

        return current_points


def training_step(num_epochs=20, num_points=250000, batch_size=128, device='cpu'):

    data_list = create_unconditional_olympic_rings(n_points=num_points, ring_thickness=0.25, verbose=False)

    train_data = data_list[:int(num_points * 0.8)]
    test_data = data_list[int(num_points * 0.8):]

    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)

    train_dataset = TensorDataset(train_data_tensor)
    test_dataset = TensorDataset(test_data_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = flow_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

    base_dist = MultivariateNormal(
        torch.zeros(model.input_dim, device=device),
        torch.eye(model.input_dim, device=device)
    )
    model.to(device)

    validation_losses = []
    training_losses = []
    log_det = []
    log_pz = []

    for epoch in range(num_epochs):
        training_loss = 0.0
        model.train()

        for i, (x,) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()

            # Transform data to base distribution
            z, log_jacobian = model.inverse(x)

            # Calculate likelihood in base distribution
            log_prob = base_dist.log_prob(z)

            # Total log likelihood includes Jacobian
            log_likelihood = log_prob + log_jacobian

            # Negative log likelihood loss
            loss = -log_likelihood.mean()

            training_loss += loss.item() * x.size(0)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        training_loss /= len(train_loader.dataset)
        training_losses.append(training_loss)


        # Validation
        model.eval()
        total_val_loss = 0.0
        log_jacob=0.0
        log_dist=0.0
        with torch.no_grad():
            for i, (x,) in enumerate(test_loader):
                x = x.to(device)
                z, log_jacobian = model.inverse(x)
                log_prob = base_dist.log_prob(z)
                val_loss = -(log_prob + log_jacobian).mean()
                total_val_loss += val_loss.item() * x.size(0)
                log_jacob += log_jacobian.mean().item() *x.size(0)
                log_dist+=log_prob.mean().item() *x.size(0)

            val_loss_avg = total_val_loss / len(test_loader.dataset)
            validation_losses.append(val_loss_avg)
            log_det.append(log_jacob)
            log_pz.append(log_dist)

            print(f"Epoch {epoch + 1} | Train Loss: {training_loss:.4f} | Val Loss: {val_loss_avg:.4f}")

        scheduler.step()

        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"normalize_flow_model_epoch_{epoch + 1}.pth")
    return model, training_losses, validation_losses,log_det,log_pz


class unconditional_matching_flow(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64,output_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.base_dist = MultivariateNormal(torch.zeros(self.input_dim).to('cpu'), torch.eye(self.input_dim).to('cpu'))
        self.nn= nn.Sequential(nn.Linear(input_dim+1, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, output_dim))

    def forward(self, y,t):
        y=torch.cat((y,t),dim=1)
        v_t = self.nn(y)
        return v_t

    def sample_progress(self, num_points=1000, device='cpu', dt=0.0001, plotting_t=[0, 0.2, 0.4, 0.6, 0.8, 1]):
        self.eval()


        _, progress_points = sample_from_matching_flow(
            self, num_points=num_points, device=device, dt=dt,
            store_trajectory=True, times=plotting_t
        )

        # Plot each set of points with different colors
        plt.figure(figsize=(8, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(progress_points)))

        for i, points in enumerate(progress_points):
            if i < len(plotting_t):
                t_val = plotting_t[i]
            else:
                t_val = 1.0  # Final time

            plt.scatter(points[:, 0], points[:, 1], s=8, color=colors[i],
                        label=f"t={t_val:.1f}", alpha=0.7)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Sample Progression Over Time")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def sample_trajectories(self, num_points=10, device='cpu', dt=0.0001):
        self.eval()

        with torch.no_grad():
            base_dist = MultivariateNormal(torch.zeros(self.input_dim).to(device),
                                           torch.eye(self.input_dim).to(device))
            y = base_dist.sample((num_points,)).to(device)

            # Store all positions over time
            all_positions = [y.cpu().numpy().copy()]

            steps = int(1 / dt)
            for i in range(steps):
                t_scalar = dt * i
                t_tensor = torch.full((num_points, 1), t_scalar, device=device)

                v_t = self.forward(y, t_tensor)
                y = y + dt * v_t


                if i % (steps // 50) == 0:  # Store ~50 snapshots
                    all_positions.append(y.cpu().numpy().copy())

            # Add final position
            all_positions.append(y.cpu().numpy().copy())


        all_positions = np.array(all_positions)  # Shape: (time_steps, num_points, 2)

        plt.figure(figsize=(10, 10))


        for particle_idx in range(num_points):
            trajectory = all_positions[:, particle_idx, :]  # Shape: (time_steps, 2)


            for t_idx in range(len(trajectory) - 1):
                color_intensity = t_idx / (len(trajectory) - 1)
                plt.plot(trajectory[t_idx:t_idx + 2, 0], trajectory[t_idx:t_idx + 2, 1],
                         color=plt.cm.viridis(color_intensity), alpha=0.7, linewidth=1)


        plt.scatter(all_positions[0, :, 0], all_positions[0, :, 1],
                    c='red', s=50, label='Start (t=0)', zorder=10, marker='o')
        plt.scatter(all_positions[-1, :, 0], all_positions[-1, :, 1],
                    c='blue', s=50, label='End (t=1)', zorder=10, marker='s')

        plt.title('Individual Particle Trajectories Through Time')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    def time_quantization(self, times=[0.002, 0.02, 0.05, 0.1, 0.2]):
        fig, axs = plt.subplots(1, len(times), figsize=(4 * len(times), 5))
        if len(times) == 1:
            axs = [axs]

        for i, dt in enumerate(times):
            # Sample with this dt
            final_points = sample_from_matching_flow(self, num_points=1000, device='cpu', dt=dt)
            points_np = final_points.cpu().numpy()

            axs[i].scatter(points_np[:, 0], points_np[:, 1], s=2, alpha=0.6)
            axs[i].set_aspect('equal', adjustable='box')
            axs[i].set_title(f'dt={dt}')
            axs[i].set_xlabel('x')
            axs[i].set_ylabel('y')
            axs[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('Effect of Different Time Steps on Final Samples', y=1.02, fontsize=16)
        plt.show()

    def reverse_flow(self, points_on_rings, points_out_of_rings, dt=0.01, device='cpu'):
        self.eval()

        # Convert inputs to tensors if needed
        if not isinstance(points_on_rings, torch.Tensor):
            points_on_rings = torch.tensor(points_on_rings, dtype=torch.float32)
        if not isinstance(points_out_of_rings, torch.Tensor):
            points_out_of_rings = torch.tensor(points_out_of_rings, dtype=torch.float32)

        points_on_rings = points_on_rings.to(device)
        points_out_of_rings = points_out_of_rings.to(device)

        # Storage for trajectories
        on_ring_trajectory = [points_on_rings.clone().cpu().numpy()]
        out_ring_trajectory = [points_out_of_rings.clone().cpu().numpy()]
        time_values = [1.0]

        # Current positions
        current_on = points_on_rings.clone()
        current_out = points_out_of_rings.clone()

        with torch.no_grad():
            steps = int(1.0 / dt)

            for step in range(steps):
                t_val = 1.0 - (step + 1) * dt  # Going backwards from 1 to 0

                # Time tensors for model input
                t_tensor_on = torch.full((current_on.size(0), 1), t_val + dt, device=device)
                t_tensor_out = torch.full((current_out.size(0), 1), t_val + dt, device=device)

                # Backward integration: y_prev = y_curr - dt * v(y_curr, t)
                v_on = self.forward(current_on, t_tensor_on) if hasattr(self, 'forward') else (
                    self(current_on, t_tensor_on)[0], None)
                v_out = self.forward(current_out, t_tensor_out) if hasattr(self, 'forward') else (
                    self(current_out, t_tensor_out)[0], None)

                current_on = current_on - dt * v_on
                current_out = current_out - dt * v_out

                # Store positions at regular intervals
                if step % max(1, steps // 10) == 0:  # Store ~10 snapshots
                    on_ring_trajectory.append(current_on.clone().cpu().numpy())
                    out_ring_trajectory.append(current_out.clone().cpu().numpy())
                    time_values.append(t_val)

        # Plot the reverse trajectories
        fig, ax = plt.subplots(figsize=(10, 8))

        # Setup colormap
        norm = mcolors.Normalize(vmin=min(time_values), vmax=max(time_values))
        cmap = cm.viridis
        colors = [cmap(norm(t)) for t in time_values]

        # Plot on-ring trajectories
        for i, (points, t_val) in enumerate(zip(on_ring_trajectory, time_values)):
            ax.scatter(points[:, 0], points[:, 1], s=20, color=colors[i],
                       marker='o', alpha=0.8, label=f'On-ring t={t_val:.1f}' if i < 3 else '')

        # Plot out-ring trajectories
        for i, (points, t_val) in enumerate(zip(out_ring_trajectory, time_values)):
            ax.scatter(points[:, 0], points[:, 1], s=20, color=colors[i],
                       marker='x', alpha=0.8, label=f'Out-ring t={t_val:.1f}' if i < 3 else '')

        # Add colorbar, specifying ax explicitly
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for compatibility
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Time t')

        # Decorations
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Reverse Flow: From Data Distribution to Base Distribution")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()


def training_unconditional_matching_flow(num_epochs=7, num_points=250000, batch_size=128, device='cpu'):
    data_list = create_unconditional_olympic_rings(num_points, ring_thickness=0.25, verbose=False)
    train_data = data_list[:int(num_points * 0.8)]
    test_data = data_list[int(num_points * 0.8):]
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    mean = train_data_tensor.mean(dim=0)
    std = train_data_tensor.std(dim=0)
    train_data_tensor = (train_data_tensor - mean) / std
    test_data_tensor = (test_data_tensor - mean) / std
    train_dataset = TensorDataset(train_data_tensor)
    test_dataset = TensorDataset(test_data_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = unconditional_matching_flow()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)  # Adjusted eta_min
    base_dist = MultivariateNormal(torch.zeros(model.input_dim).to(device), torch.eye(model.input_dim).to(device))
    model.to(device)
    validation_losses = []
    training_losses = []
    for epoch in range(num_epochs):
        # Reset training loss for this epoch
        training_loss = 0.0
        model.train()

        for i, (x,) in enumerate(train_loader):
            x = x.to(device)
            eps = base_dist.sample((x.size(0),))
            t = torch.rand(x.size(0), 1).to(device)
            y=t*x+(1-t)*eps
            v_t = model.forward(y,t)
            loss = ((v_t - (x-eps)) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss+= loss.item() * x.size(0)
        # Average training loss for this epoch
        training_loss /= len(train_loader.dataset)
        training_losses.append(training_loss)
        # Validation step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for i, (x,) in enumerate(test_loader):
                x = x.to(device)
                eps = base_dist.sample((x.size(0),))
                t = torch.rand(x.size(0), 1).to(device)
                y=t*x+(1-t)*eps
                v_t = model.forward(y,t)
                val_loss = ((v_t - (x-eps)) ** 2).mean()
                total_val_loss += val_loss.item() * x.size(0)
            val_loss_avg = total_val_loss / len(test_loader.dataset)
            validation_losses.append(val_loss_avg)
            print(f"Epoch {epoch + 1} | Train Loss: {training_loss:.4f} | Val Loss: {val_loss_avg:.4f}")
            torch.save(model.state_dict(), f"unconditional_matching_flow model_epoch_{epoch + 1}.pth")
        scheduler.step()
    # samples = sample_from_matching_flow(model, num_points=2000, device='cpu',dt=0.0001)
    # visualize_matching_flow_samples(samples, title='Samples from Matching Flow')
    return model, training_losses, validation_losses

class conditional_model(nn.Module):
      def __init__(self, input_dim=2, hidden_dim=64,output_dim=2,label_dim=5,label_embed_dim=16):
          super().__init__()
          self.input_dim = input_dim
          self.hidden_dim = hidden_dim
          self.label_embedding = nn.Embedding(num_embeddings=label_dim, embedding_dim=label_embed_dim)
          self.nn= nn.Sequential(nn.Linear(input_dim+1+label_embed_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LeakyReLU(),
                                             nn.Linear(hidden_dim, output_dim))

      def forward(self, y,label,t):
          label_embed =self.label_embedding(label)
          y=torch.cat((y,t,label_embed),dim=1)
          v_t = self.nn(y)
          return v_t

      def sample_from_each_class(self, dt=0.0001, device='cpu'):
          self.eval()

          points_all, labels_all = sample_olympic_rings(num_points_per_ring=1)
          points_to_reverse = []
          idx = 0
          while len(found_colors) < 5 and idx < len(points_all):
              pt = points_all[idx]
              color = labels_all[idx]
              if color not in found_colors:
                  points_to_reverse.append((pt, color))
                  found_colors.add(color)
              idx += 1

          if len(points_to_reverse) < 5:
              raise RuntimeError("Could not find one point of each color in the sample.")


          raw_points, raw_colors = zip(*points_to_reverse)
          color_to_int = {
              'blue': 0,
              'black': 1,
              'red': 2,
              'yellow': 3,
              'green': 4
          }
          label_indices = torch.tensor(
              [color_to_int[c] for c in raw_colors],
              dtype=torch.long,
              device=device)

          y = torch.tensor(raw_points, dtype=torch.float32, device=device)
          steps = int(1.0 / dt)
          snapshots = 50
          interval = max(1, steps // snapshots)

          all_positions = [y.detach().cpu().numpy().copy()]
          time_values = [1.0]

          current_y = y.clone()

          with torch.no_grad():
              for step in range(steps):
                  t_curr = 1.0 - step * dt
                  t_next = t_curr - dt


                  t_tensor = torch.full((current_y.size(0), 1), t_curr, device=device)
                  v = self.forward(current_y, label_indices,t_tensor)
                  current_y = current_y - dt * v
                  if (step + 1) % interval == 0:
                      all_positions.append(current_y.detach().cpu().numpy().copy())
                      time_values.append(t_next)


          if len(all_positions) < snapshots + 1:
              all_positions.append(current_y.detach().cpu().numpy().copy())
              time_values.append(0.0)

          all_positions = np.stack(all_positions, axis=0)
          time_values = np.array(time_values)
          plt.figure(figsize=(10, 10))

          num_snaps = all_positions.shape[0]
          num_pts = all_positions.shape[1]
          particle_colors = raw_colors

          for pt_idx in range(num_pts):
              traj = all_positions[:, pt_idx, :]
              c = particle_colors[pt_idx]


              plt.plot(traj[:, 0], traj[:, 1], '-', color=c, linewidth=1.5, alpha=0.8)


              plt.scatter(
                  traj[0, 0], traj[0, 1],
                  color=c, marker='o', s=80, edgecolor='white',
                  label=f"{c} start" if pt_idx == 0 else ""
              )
              plt.scatter(
                  traj[-1, 0], traj[-1, 1],
                  color=c, marker='X', s=80, edgecolor='black',
                  label=f"{c} end" if pt_idx == 0 else ""
              )

          plt.title("Reverse‐Flow Trajectories of 5 Points (One from Each Ring Color)")
          plt.xlabel("x")
          plt.ylabel("y")
          plt.gca().set_aspect('equal', adjustable='box')
          plt.grid(True, alpha=0.3)
          legend_handles = [
              Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='Start'),
              Line2D([], [], color='gray', marker='X', linestyle='None', markersize=10, label='End'),
          ]
          plt.legend(handles=legend_handles, loc='upper right')

          plt.tight_layout()
          plt.show()

      def sample_from_conditional_matching_flow(self, num_points_per_label=600, device='cpu', dt=0.0001):
          model.eval()
          all_points = []
          all_labels = []


          for label in range(5):
              with torch.no_grad():
                  # Sample from base distribution
                  base_dist = MultivariateNormal(
                      torch.zeros(model.input_dim).to(device),
                      torch.eye(model.input_dim).to(device)
                  )
                  y = base_dist.sample((num_points_per_label,)).to(device)


                  label_tensor = torch.full((num_points_per_label,), label, dtype=torch.long, device=device)


                  steps = int(1 / dt)
                  for i in range(steps):
                      t_scalar = dt * i
                      t_tensor = torch.full((num_points_per_label, 1), t_scalar, device=device)


                      v_t = model.forward(y,label_tensor, t_tensor )

                      # Update positions
                      y = y + dt * v_t

                  # Store results
                  all_points.append(y.cpu().numpy())
                  all_labels.extend([label] * num_points_per_label)

          all_points = np.vstack(all_points)
          all_labels = np.array(all_labels)

          return all_points, all_labels

      def sample_and_plot_conditional(self,
                                      data_mean: np.ndarray,
                                      data_std: np.ndarray,
                                      num_per_label: int = 1000,
                                      dt: float = 0.0001,
                                      device: str = 'cpu'):

          self.eval()
          self.to(device)

          base_dist = torch.distributions.MultivariateNormal(
              torch.zeros(self.input_dim, device=device),
              torch.eye(self.input_dim, device=device)
          )

          int_to_color = {
              0: 'black',
              1: 'blue',
              2: 'green',
              3: 'red',
              4: 'yellow'
          }
          label_names = {
              0: 'Blue Ring',
              1: 'Black Ring',
              2: 'Red Ring',
              3: 'Yellow Ring',
              4: 'Green Ring'
          }

          steps = int(1.0 / dt)

          plt.figure(figsize=(10, 10))

          for ℓ in range(5):
              # 1) Noise in normalized space
              y = base_dist.sample((num_per_label,)).to(device)

              # 2) Integrate forward under v(y,ℓ,t)
              for step in range(steps):
                  t_curr = step * dt
                  t_tensor = torch.full((num_per_label, 1), t_curr, device=device)
                  label_tensor = torch.full((num_per_label,), ℓ, dtype=torch.long, device=device)

                  # Correct argument order: forward(y, label, t)
                  v = self.forward(y, label_tensor, t_tensor)
                  y = y + dt * v
              y_norm = y.detach().cpu().numpy()
              plt.scatter(
                  y_norm[:, 0],
                  y_norm[:, 1],
                  s=5,
                  alpha=0.6,
                  color=int_to_color[ℓ],
                  label=label_names[ℓ]
              )

          plt.title("Conditional Matching Flow Samples (Normalized Space)", fontsize=16)
          plt.xlabel("x (normalized)", fontsize=14)
          plt.ylabel("y (normalized)", fontsize=14)
          plt.legend(loc='upper right', fontsize=12)
          plt.grid(True, alpha=0.3)
          plt.gca().set_aspect('equal', adjustable='box')
          plt.tight_layout()
          plt.show()


def train_cond(num_epochs=20, num_points=250000, batch_size=128, device='cpu'):
    data_points, labels, int_to_label = create_olympic_rings(
        num_points, ring_thickness=0.25, verbose=False
    )
    perm = np.random.permutation(num_points)
    data_points = data_points[perm]
    labels      = labels[perm]

    split_idx     = int(0.8 * num_points)
    train_data_np = data_points[:split_idx]
    train_labels  = labels[:split_idx]
    test_data_np  = data_points[split_idx:]
    test_labels   = labels[split_idx:]


    train_data_tensor  = torch.tensor(train_data_np, dtype=torch.float32, device=device)
    train_labels_tensor= torch.tensor(train_labels, dtype=torch.long,  device=device)
    test_data_tensor   = torch.tensor(test_data_np, dtype=torch.float32, device=device)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long,  device=device)

    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    test_dataset  = TensorDataset(test_data_tensor,  test_labels_tensor)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    model = conditional_model()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)

    base_dist = MultivariateNormal(
        torch.zeros(model.input_dim, device=device),
        torch.eye(model.input_dim,  device=device)
    )

    training_losses = []
    validation_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        for x_norm, lab in train_loader:
            eps = base_dist.sample((x_norm.size(0),))
            t   = torch.rand(x_norm.size(0), 1, device=device)
            y   = t * x_norm + (1.0 - t) * eps

            v_pred  = model.forward(y, lab, t)
            v_target= x_norm - eps
            loss    = ((v_pred - v_target).pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * x_norm.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        training_losses.append(avg_train_loss)
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_norm, lab in test_loader:
                eps = base_dist.sample((x_norm.size(0),))
                t   = torch.rand(x_norm.size(0), 1, device=device)
                y   = t * x_norm + (1.0 - t) * eps

                v_pred   = model.forward(y, lab, t)
                v_target = x_norm - eps
                val_loss = ((v_pred - v_target).pow(2)).mean()

                total_val_loss += val_loss.item() * x_norm.size(0)

        avg_val_loss = total_val_loss / len(test_loader.dataset)
        validation_losses.append(avg_val_loss)

        print(f"Epoch {epoch:02d}/{num_epochs:02d} | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        scheduler.step()
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       f"conditional_matching_flow_model_epoch_{epoch}.pth")
    return model, np.array([0.0, 0.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)



if __name__ == "__main__":
    # Example usage
    num_points = 250000
    seeds = [42,108,555]
    device = 'cpu'
    points_on_rings = create_unconditional_olympic_rings(3, ring_thickness=0.25, verbose=False)
    points_out_of_rings = MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device)).sample((2,)).to(device)
    if TRAIN_NORMALLIZE:
        model,training_losses,validation_losses,log_det,log_pz = training_step(num_epochs=20)
        plot_validation_loss(training_losses, validation_losses,log_det,log_pz) #Q1
        model = flow_model()
        model.load_state_dict(torch.load(f"normalize_flow_model_epoch_20.pth", map_location=device))
        visualize_model() #Q2
        model.sample_over_time() #Q3
        model.sample_trajectories() #Q4
        model.sample_inverse(points_on_rings,points_out_of_rings) #Q5
    if TRAIN_UNCONDITONAL:
        model,training_losses,validation_losses = training_unconditional_matching_flow(num_epochs=20)
        plot_validation_loss(training_losses, validation_losses)#Q1
        model = unconditional_matching_flow()
        model.load_state_dict(torch.load(f"unconditional_matching_flow model_epoch_20.pth", map_location=device))
        model.sample_progress(num_points=1000, device='cpu', dt=0.0001, plotting_t=[0, 0.2, 0.4, 0.6, 0.8, 1]) #Q2
        model.sample_trajectories(num_points=10, device='cpu', dt=0.0001) #Q3
        model.time_quantization(times=[0.002, 0.02, 0.05, 0.1, 0.2]) #Q4
        points = model.reverse_flow(points_on_rings,points_out_of_rings) #Q5

    # if TRAIN_CONDITIONAL:
    model, data_mean, data_std = train_cond(
        num_epochs=20,
        num_points=250000,
        batch_size=128,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model = conditional_model()
    model.load_state_dict(torch.load(f"conditional_matching_flow_model_epoch_20.pth", map_location=device))
    model.sample_from_each_class()#Q2
    points,labels = model.sample_from_conditional_matching_flow()
    model.sample_and_plot_conditional(
        data_mean=data_mean,
        data_std=data_std,
        num_per_label=600,
        dt=1e-4,
        device=device
    )
