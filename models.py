import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch import optim
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import shutil
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
from torchvision.datasets import MNIST
import random
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score




# Create stronger augmentations as per VICReg requirements
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])


class Encoder(nn.Module):
    def __init__(self, D=128, device='cuda'):
        super(Encoder, self).__init__()
        # Fix: Remove pretrained parameter (deprecated)
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, 512)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, D)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def encode(self, x):
        return self.forward(x)


class Projector(nn.Module):
    def __init__(self, D, proj_dim=512):
        super(Projector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(D, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.model(x)


class Classifier(nn.Module):
    def __init__(self, input_num=128, output_num=10):
        super(Classifier, self).__init__()
        self.layer = nn.Linear(input_num, output_num)

    def forward(self, x):
        return self.layer(x)


def load_data():
    # Create data directory
    os.makedirs('./data', exist_ok=True)

    # Base transform for loading data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load CIFAR-10 (will download automatically if not present)
    train_dataset = CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # PDF specifies batch size 256, use num_workers=0 for Windows compatibility
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader


def augment_batch(images, transform, device='cuda'):
    augmented_images = []

    for i in range(images.shape[0]):
        single_image = images[i]
        # Apply transform
        augmented = transform(single_image)
        augmented_images.append(augmented)

    # Stack back into a batch
    return torch.stack(augmented_images).to(device)


def variance_loss(z_i, z_j, eps=1e-4, gamma=1.0):
    def _branch(z):
        # 1) center
        z_centered = z - z.mean(dim=0, keepdim=True)  # (B, d)
        # 2) population variance (denominator = B)
        var = (z_centered ** 2).mean(dim=0)  # (d,)
        # 3) std + hinge
        std = torch.sqrt(var + eps)  # (d,)
        return torch.mean(F.relu(gamma - std))  # scalar

    return _branch(z_i) + _branch(z_j)


def cov_loss(z_i, z_j):
    def _branch(z):
        B, d = z.shape
        # 1) center
        z_centered = z - z.mean(dim=0, keepdim=True)  # (B, d)
        # 2) unbiased covariance
        cov = (z_centered.T @ z_centered) / (B - 1)  # (d, d)
        # 3) sum squared off‐diagonal entries
        mask = ~torch.eye(d, dtype=bool, device=z.device)
        off_diag = cov[mask]
        return torch.sum(off_diag ** 2) / d  # scalar

    return _branch(z_i) + _branch(z_j)


def train_VICReg_regular(encoder, projector, train_loader, test_loader,
                         device='cuda', gamma=1, lmbda=25, mu=25, nu=1, eps=1e-4, num_epochs=30):
    if not torch.cuda.is_available():
        device = 'cpu'

    encoder.to(device)
    projector.to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-6
    )

    mse_loss = nn.MSELoss()
    inv_loss = []
    var_loss = []
    cov_loss_value = []
    test_loss = []

    for epoch in range(num_epochs):
        # Training phase
        encoder.train()
        projector.train()

        epoch_inv, epoch_var, epoch_cov = 0, 0, 0

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)

            # Forward pass with two different augmentations
            y_i = encoder(augment_batch(images, train_transform, device))
            y_j = encoder(augment_batch(images, train_transform, device))

            z_i = projector(y_i)
            z_j = projector(y_j)

            batch_inv = lmbda * mse_loss(z_i, z_j)

            batch_var = mu * variance_loss(z_i, z_j, eps, gamma)

            batch_cov = nu * cov_loss(z_i, z_j)

            total_loss = batch_inv + batch_var + batch_cov

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            epoch_inv += batch_inv.item() / lmbda
            epoch_var += batch_var.item() / mu
            epoch_cov += batch_cov.item() / nu

        # Store epoch averages
        inv_loss.append(epoch_inv / len(train_loader))
        var_loss.append(epoch_var / len(train_loader))
        cov_loss_value.append(epoch_cov / len(train_loader))

        print(f'TRAIN Epoch {epoch + 1}/{num_epochs}: '
              f'Inv: {inv_loss[-1]:.4f}, Var: {var_loss[-1]:.4f}, Cov: {cov_loss_value[-1]:.4f}')

        # Evaluation phase
        encoder.eval()
        projector.eval()
        test_epoch_loss = 0

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                y_i = encoder(augment_batch(images, test_transform, device))
                y_j = encoder(augment_batch(images, test_transform, device))

                z_i = projector(y_i)
                z_j = projector(y_j)

                batch_inv = lmbda * mse_loss(z_i, z_j)
                batch_var = mu * variance_loss(z_i, z_j, eps, gamma)
                batch_cov = nu * cov_loss(z_i, z_j)

                test_epoch_loss += (batch_inv / lmbda + batch_var / mu + batch_cov / nu).item()

        test_loss.append(test_epoch_loss / len(test_loader))
        print(f'TEST Epoch {epoch + 1}/{num_epochs}: Loss: {test_loss[-1]:.4f}')

        # Save models every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(encoder.state_dict(), f'./results/encoder_{epoch + 1}.pth')
            torch.save(projector.state_dict(), f'./results/projector_{epoch + 1}.pth')
            print(f"💾 Models saved at epoch {epoch + 1}")

    # Save final models
    torch.save(encoder.state_dict(), './results/encoder_30.pth')
    torch.save(projector.state_dict(), './results/projector_30.pth')

    return inv_loss, var_loss, cov_loss_value, test_loss


def train_VICReg_ablation(encoder, projector, train_loader, test_loader,
                          device='cuda', gamma=1, lmbda=25, mu=0, nu=1, eps=1e-4, num_epochs=30):
    # Check device availability
    if not torch.cuda.is_available():
        device = 'cpu'

    encoder.to(device)
    projector.to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-6
    )

    mse_loss = nn.MSELoss()
    inv_loss = []
    var_loss = []
    cov_loss_value = []
    test_loss = []

    # Create results directory
    os.makedirs('./results', exist_ok=True)

    for epoch in range(num_epochs):
        # Training phase
        encoder.train()
        projector.train()

        epoch_inv, epoch_var, epoch_cov = 0, 0, 0

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)

            # Forward pass with two different augmentations
            y_i = encoder(augment_batch(images, train_transform, device))
            y_j = encoder(augment_batch(images, train_transform, device))

            z_i = projector(y_i)
            z_j = projector(y_j)

            batch_inv = lmbda * mse_loss(z_i, z_j)

            batch_var = mu * variance_loss(z_i, z_j, eps, gamma)

            batch_cov = nu * cov_loss(z_i, z_j)

            total_loss = batch_inv + batch_var + batch_cov

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            epoch_inv += batch_inv.item()
            epoch_var += batch_var.item()
            epoch_cov += batch_cov.item()

        # Store epoch averages
        inv_loss.append(epoch_inv / len(train_loader))
        var_loss.append(epoch_var / len(train_loader))
        cov_loss_value.append(epoch_cov / len(train_loader))

        print(f'TRAIN Epoch {epoch + 1}/{num_epochs}: '
              f'Inv: {inv_loss[-1]:.4f}, Var: {var_loss[-1]:.4f}, Cov: {cov_loss_value[-1]:.4f}')

        # Evaluation phase
        encoder.eval()
        projector.eval()
        test_epoch_loss = 0

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)

                y_i = encoder(augment_batch(images, test_transform, device))
                y_j = encoder(augment_batch(images, test_transform, device))

                z_i = projector(y_i)
                z_j = projector(y_j)

                batch_inv = lmbda * mse_loss(z_i, z_j)
                batch_var = mu * variance_loss(z_i, z_j, eps, gamma)
                batch_cov = nu * cov_loss(z_i, z_j)

                test_epoch_loss += (batch_inv + batch_var + batch_cov).item()

        test_loss.append(test_epoch_loss / len(test_loader))
        print(f'TEST Epoch {epoch + 1}/{num_epochs}: Loss: {test_loss[-1]:.4f}')

        # Save models every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(encoder.state_dict(), f'./results/encoder_abl1_{epoch + 1}.pth')
            torch.save(projector.state_dict(), f'./results/projector_abl1_{epoch + 1}.pth')
            print(f"💾 Models saved at epoch {epoch + 1}")

    # Save final models
    torch.save(encoder.state_dict(), './results/encoder_abl1_30.pth')
    torch.save(projector.state_dict(), './results/projector_abl1_30.pth')

    return inv_loss, var_loss, cov_loss_value, test_loss


def find_neighbors(loader, device=None, save_path='./results/neighbor_mapping_train.pkl'):
    """
    Find nearest neighbors in embedding space for each training sample
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"🔄 Finding nearest neighbors on device: {device}")

    # Load pre-trained encoder (from first ablation)
    encoder = Encoder()
    model_path = './results/encoder_30.pth'

    if device == 'cpu':
        encoder.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        encoder.load_state_dict(torch.load(model_path))

    encoder.to(device)
    encoder.eval()

    all_embeddings = []
    all_indices = []  # Keep track of original indices

    with torch.no_grad():
        global_idx = 0
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            batch_embeddings = encoder(images)
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings_np)

            # Store global indices for this batch
            batch_indices = list(range(global_idx, global_idx + len(images)))
            all_indices.extend(batch_indices)
            global_idx += len(images)

            if (batch_idx + 1) % 50 == 0:
                print(f"   Processed {batch_idx + 1} batches...")

    embeddings = np.concatenate(all_embeddings, axis=0)
    nbrs = NearestNeighbors(n_neighbors=4, metric='euclidean')
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Create mapping: index -> neighbor index (skip self at index 0)
    top3_indices = indices[:, 1:4]  # Skip self-similarity
    neighbor_dict = {}

    for i in range(len(embeddings)):
        # Randomly choose one of the top 3 neighbors
        chosen_neighbor_idx = random.choice(top3_indices[i])
        neighbor_dict[i] = int(chosen_neighbor_idx)

    # Save neighbor mapping
    os.makedirs('./results', exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(neighbor_dict, f)

    return neighbor_dict, encoder


def train_VICReg_abl2(train_loader, projector, neighbor_dict, device=None,
                      gamma=1, ld=25, mu=25, v=1, eps=1e-4, num_epochs=5):
    """
    Train VICReg using nearest neighbors instead of augmentations (Ablation 2)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained encoder
    encoder = Encoder()
    encoder_path = './results/encoder_abl1_30.pth'

    if device == 'cpu':
        encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    else:
        encoder.load_state_dict(torch.load(encoder_path))

    encoder.to(device)
    projector.to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-6
    )

    encoder.train()
    projector.train()
    mse_loss = nn.MSELoss()

    dataset_list = []
    for images, labels in train_loader:
        for i in range(len(images)):
            dataset_list.append(images[i])

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        num_batches = 0

        # Create batches with neighbor pairs
        batch_size = 128
        indices = list(range(len(dataset_list)))
        random.shuffle(indices)

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]

            # Get images and their neighbors
            batch_images_i = []
            batch_images_j = []

            for idx in batch_indices:
                if idx in neighbor_dict and neighbor_dict[idx] < len(dataset_list):
                    batch_images_i.append(dataset_list[idx])
                    batch_images_j.append(dataset_list[neighbor_dict[idx]])

            if len(batch_images_i) < 2:  # Skip if batch too small
                continue

            # Stack into tensors
            images_i = torch.stack(batch_images_i).to(device)
            images_j = torch.stack(batch_images_j).to(device)

            # Forward pass
            y_i = encoder(images_i)
            y_j = encoder(images_j)

            z_i = projector(y_i)
            z_j = projector(y_j)

            # Compute loss
            batch_inv = ld * mse_loss(z_i, z_j)
            batch_var = mu * variance_loss(z_i, z_j, eps, gamma)
            batch_cov = v * cov_loss(z_i, z_j)

            total_loss = batch_inv + batch_var + batch_cov

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                print(f"   Batch {num_batches}: Loss = {total_loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    # Save models
    torch.save(encoder.state_dict(), './results/encoder_abl2_30.pth')
    torch.save(projector.state_dict(), './results/projector_abl2_30.pth')

    return encoder


def run_ablation2_training(train_loader, test_loader):
    # Step 1: Find neighbors
    neighbor_dict, _ = find_neighbors(train_loader)
    projector = Projector(D=128)
    projector_path = './results/projector_30.pth'

    encoder = train_VICReg_abl2(train_loader, projector, neighbor_dict, num_epochs=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier, final_accuracy = train_classifier(
        train_loader, test_loader,
        device=device,
        num_epochs=10,
        model_path='./results/encoder_abl2_30.pth'
    )

    return encoder, classifier


def denormalize_cifar(tensor):
    """
    Denormalize CIFAR-10 images for visualization
    """
    # CIFAR-10 normalization: mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)

    # Denormalize: x = x * std + mean
    denorm = tensor * std + mean

    # Clamp to [0, 1] and convert to numpy
    denorm = torch.clamp(denorm, 0, 1)
    return denorm.permute(1, 2, 0).numpy()


def select_random_img(train_loader):
    class_samples = {}
    class_indices = {}

    for batch_idx, (images, labels) in enumerate(train_loader):
        for i, (img, label) in enumerate(zip(images, labels)):
            class_id = label.item()
            if class_id not in class_samples:
                class_samples[class_id] = img
                class_indices[class_id] = batch_idx * train_loader.batch_size + i
            if len(class_samples) == 10:
                break
        if len(class_samples) == 10:
            break

    return class_samples, class_indices


def encode_dataset(encoder, train_loader, device='cuda'):
    encoder.eval()
    encoder.to(device)
    all_features = []
    all_images = []

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            features = encoder(images)
            all_features.append(features.cpu())
            all_images.append(images.cpu())

            if (batch_idx + 1) % 100 == 0:
                print(f"   Processed {batch_idx + 1} batches...")

    all_features = torch.cat(all_features, dim=0)
    all_images = torch.cat(all_images, dim=0)

    return all_features, all_images


def find_neighbors_vis(sample_features, all_features, k=5):
    """
    Find k nearest and k farthest neighbors in embedding space
    """
    sample_np = sample_features.cpu().numpy()
    all_np = all_features.cpu().numpy()

    # Compute Euclidean distances
    distances = np.sqrt(np.sum((all_np - sample_np) ** 2, axis=1))
    sorted_indices = np.argsort(distances)

    # Skip the first one (self) for nearest neighbors
    nearest_idx = sorted_indices[1:k + 1]

    # Get farthest neighbors (reverse order)
    farthest_idx = sorted_indices[-k:][::-1]

    return nearest_idx, farthest_idx


def plot_all_neighbors_grid(class_samples, all_nearest, all_farthest, neighbor_type, model_name):
    """
    Plot all classes in a single grid figure (10 rows x 6 columns)
    neighbor_type: 'Nearest' or 'Farthest'
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Create large figure with 10 rows, 6 columns
    fig, axes = plt.subplots(10, 6, figsize=(18, 25))

    # Create informative title
    title = f"{neighbor_type} Neighbors Analysis - {model_name}\n"
    title += f"Each row: Original image + 5 {neighbor_type.lower()} neighbors in embedding space"
    fig.suptitle(title, fontsize=16, y=0.98, weight='bold')

    # Process each class (row)
    for class_id in range(10):
        if class_id in class_samples:
            sample_img = class_samples[class_id]

            # Plot original image (column 0)
            sample_np = denormalize_cifar(sample_img)
            axes[class_id, 0].imshow(sample_np)
            axes[class_id, 0].set_title(f'{class_names[class_id]}\n(Original)',
                                        fontsize=10, weight='bold')
            axes[class_id, 0].axis('off')

            # Add colored border to original
            for spine in axes[class_id, 0].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('blue' if neighbor_type == 'Nearest' else 'red')
                spine.set_linewidth(3)

            # Plot neighbor images (columns 1-5)
            neighbors = all_nearest[class_id] if neighbor_type == 'Nearest' else all_farthest[class_id]

            for i, neighbor_img in enumerate(neighbors):
                neighbor_np = denormalize_cifar(neighbor_img)
                axes[class_id, i + 1].imshow(neighbor_np)
                axes[class_id, i + 1].set_title(f'{neighbor_type} {i + 1}', fontsize=9)
                axes[class_id, i + 1].axis('off')
        else:
            # Handle missing class
            for j in range(6):
                axes[class_id, j].text(0.5, 0.5, 'No sample',
                                       ha='center', va='center', fontsize=12)
                axes[class_id, j].set_title(f'{class_names[class_id]}\n(Missing)', fontsize=10)
                axes[class_id, j].axis('off')

    # Add column labels
    column_labels = ['Original'] + [f'{neighbor_type} {i + 1}' for i in range(5)]
    for j, label in enumerate(column_labels):
        axes[0, j].text(0.5, 1.15, label, transform=axes[0, j].transAxes,
                        ha='center', va='bottom', fontsize=11, weight='bold')

    # Add row labels (class numbers)
    for i in range(10):
        axes[i, 0].text(-0.1, 0.5, f'Class {i}', transform=axes[i, 0].transAxes,
                        ha='right', va='center', fontsize=11, weight='bold', rotation=90)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, left=0.05, right=0.98)

    # Save figure
    os.makedirs('./results', exist_ok=True)
    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    filename = f'./results/{neighbor_type.lower()}_neighbors_{safe_model_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


def sample_and_compare_models(train_loader):
    # Check which models are available
    model_paths = {
        'Regular Encoder': './results/encoder_30.pth',
        'Ablation 2 (Neighbors)': './results/encoder_abl2_30.pth'
    }

    available_models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            available_models[name] = path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_samples, class_indices = select_random_img(train_loader)

    # Process each available model
    for model_name, model_path in available_models.items():

        # Load encoder
        encoder = Encoder()
        if device == 'cpu':
            encoder.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            encoder.load_state_dict(torch.load(model_path))

        # Encode dataset
        all_features, all_images = encode_dataset(encoder, train_loader, device)

        # Collect all neighbors for grid plotting
        all_nearest_neighbors = {}
        all_farthest_neighbors = {}

        # Process each class
        for class_id in sorted(class_samples.keys()):
            sample_img = class_samples[class_id]
            sample_idx = class_indices[class_id]
            sample_features = all_features[sample_idx]

            # Find neighbors
            nearest_idx, farthest_idx = find_neighbors_vis(sample_features, all_features)

            # Store neighbor images
            all_nearest_neighbors[class_id] = [all_images[idx] for idx in nearest_idx]
            all_farthest_neighbors[class_id] = [all_images[idx] for idx in farthest_idx]

        # Plot nearest neighbors grid
        plot_all_neighbors_grid(class_samples, all_nearest_neighbors, all_farthest_neighbors,
                                'Nearest', model_name)

        # Plot farthest neighbors grid
        plot_all_neighbors_grid(class_samples, all_nearest_neighbors, all_farthest_neighbors,
                                'Farthest', model_name)


def run_neighbor_sampling(train_loader):
    """
    Wrapper function for running neighbor sampling analysis
    """
    try:
        # Get available models
        model_paths = {
            'Regular VICReg': './results/encoder_30.pth',
            'VICReg with Neighbors': './results/encoder_abl2_30.pth'
        }

        available_models = []
        for name, path in model_paths.items():
            if os.path.exists(path):
                available_models.append((name, path))

        if not available_models:
            print("❌ No trained models found for neighbor sampling!")
            print("💡 Please train models first (options 1 and 5)")
            return

        print("🖼️ Starting neighbor sampling analysis...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for model_name, model_path in available_models:
            print(f"\n🔍 Analyzing {model_name}...")
            all_images, all_features = sample_and_compare_models(train_loader)
            print(f"✅ Completed neighbor analysis for {model_name}")

        print("🎉 Neighbor sampling analysis complete!")

    except Exception as e:
        print(f"❌ Neighbor sampling failed: {e}")
        print("💡 Common issues:")
        print("   - No trained models available")
        print("   - Insufficient memory for encoding large dataset")
        print("   - CUDA/CPU compatibility issues")


def plot_losses(inv_loss, var_loss, cov_loss_value, test_loss, num_epochs):
    """Plot training losses over epochs"""
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, inv_loss, label='Invariant Loss', marker='o', linewidth=2)
    plt.plot(epochs, var_loss, label='Variance Loss (μ=0)', marker='s', linewidth=2)
    plt.plot(epochs, cov_loss_value, label='Covariance Loss', marker='^', linewidth=2)
    plt.plot(epochs, test_loss, label='Total Test Loss', marker='d', linewidth=2)

    plt.title('VICReg Ablation Study: Training Without Variance Term (μ=0)', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig('./results/ablation_losses.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_regular_losses(inv_loss, var_loss, cov_loss_value, test_loss, num_epochs):
    """Plot training losses for regular VICReg training"""
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, inv_loss, label='Invariant Loss', marker='o', linewidth=2)
    plt.plot(epochs, var_loss, label='Variance Loss', marker='s', linewidth=2)
    plt.plot(epochs, cov_loss_value, label='Covariance Loss', marker='^', linewidth=2)
    plt.plot(epochs, test_loss, label='Total Test Loss', marker='d', linewidth=2)

    plt.title('VICReg Standard Training: All Loss Components', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig('./results/regular_training_losses.png', dpi=300, bbox_inches='tight')
    plt.show()


def two_d_reduction(test_loader, model_path='./results/encoder_abl1_30.pth', device='cuda'):
    # Load trained encoder
    encoder = Encoder()
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    features = []
    labels = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            batch_features = encoder(images)
            features.append(batch_features.cpu().numpy())
            labels.extend(targets.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    # PCA
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(features)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_2d = tsne.fit_transform(features)

    print("✅ 2D reduction complete")
    return pca_2d, tsne_2d, labels


def plot_2d(pca_2d, tsne_2d, colors):
    """Plot PCA and t-SNE visualizations with properly positioned legend"""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # PCA plot
    scatter1 = ax1.scatter(pca_2d[:, 0], pca_2d[:, 1], c=colors, cmap='tab10', alpha=0.6, s=1)
    ax1.set_title('PCA - VICReg', fontsize=14, pad=15)
    ax1.set_xlabel('First Principal Component', fontsize=12)
    ax1.set_ylabel('Second Principal Component', fontsize=12)

    # t-SNE plot
    scatter2 = ax2.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=colors, cmap='tab10', alpha=0.6, s=1)
    ax2.set_title('t-SNE - VICReg ', fontsize=14, pad=15)
    ax2.set_xlabel('t-SNE 1', fontsize=12)
    ax2.set_ylabel('t-SNE 2', fontsize=12)

    # Adjust subplot positioning to make room for colorbar
    plt.subplots_adjust(left=0.08, right=0.82, top=0.9, bottom=0.1, wspace=0.3)

    # Add colorbar with proper positioning (further to the right)
    cbar_ax = fig.add_axes([0.84, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter1, cax=cbar_ax)
    cbar.set_ticks(range(10))
    cbar.set_ticklabels(class_names, fontsize=11)
    cbar.set_label('CIFAR-10 Classes', fontsize=12, labelpad=15)

    plt.savefig('./results/2d_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def train_classifier(train_loader, test_loader, device='cuda', num_epochs=10,
                     model_path='./results/encoder_30.pth'):
    encoder = Encoder()
    if device == 'cuda':
        encoder.load_state_dict(torch.load(model_path, map_location='cuda'))
    else:
        encoder.load_state_dict(torch.load(model_path))

    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    classifier = Classifier(input_num=128, output_num=10)
    classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-6)

    for epoch in range(num_epochs):
        classifier.train()
        train_accuracy = 0
        train_total = 0
        train_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                features = encoder(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            train_accuracy += (predicted == labels).sum().item()
            train_loss += loss.item()
            train_total += labels.size(0)
        epoch_train_acc = train_accuracy * 100 / train_total
        avg_train_loss = train_loss / len(train_loader)

        print(f'TRAIN Epoch {epoch + 1}/{num_epochs}: Loss: {avg_train_loss:.4f}, Accuracy: {epoch_train_acc:.2f}%')

        classifier.eval()
        test_accuracy = 0
        test_total = 0
        test_loss = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features = encoder(images)
                outputs = classifier(features)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                test_accuracy += (predicted == labels).sum().item()
                test_loss += loss.item()
                test_total += labels.size(0)
        epoch_test_acc = test_accuracy * 100 / test_total
        avg_test_loss = test_loss / len(test_loader)

        print(f'TEST Epoch {epoch + 1}/{num_epochs}: Loss: {avg_test_loss:.4f}, Accuracy: {epoch_test_acc:.2f}%')

    print(f"✅ Training complete! Final test accuracy: {epoch_test_acc:.2f}%")

    # Save classifier
    os.makedirs('./results', exist_ok=True)
    torch.save(classifier.state_dict(), './results/classifier.pth')

    return classifier, epoch_test_acc


# ==================== ANOMALY DETECTION FUNCTIONS ====================

class MNISTto3Channel:
    """Transform MNIST to 3-channel to match CIFAR-10"""

    def __call__(self, img):
        # Convert grayscale to 3-channel by repeating
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img


def load_anomaly_data(batch_size=256, num_workers=4):
    # Create data directory
    os.makedirs('./data', exist_ok=True)

    # CIFAR-10 normalization
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # MNIST transform: resize to 32x32, convert to 3-channel, normalize like CIFAR-10
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        MNISTto3Channel(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Load CIFAR-10 train set (normal data)
    train_dataset = CIFAR10(
        root='./data', train=True, download=True, transform=cifar_transform
    )

    # Load CIFAR-10 test set (normal data)
    cifar_test = CIFAR10(
        root='./data', train=False, download=True, transform=cifar_transform
    )

    # Load MNIST test set (anomaly data)
    mnist_test = MNIST(
        root='./data', train=False, download=True, transform=mnist_transform
    )

    class AnomalyTestDataset:
        def __init__(self, cifar_dataset, mnist_dataset):
            self.cifar_data = []
            self.mnist_data = []
            self.labels = []

            # Add CIFAR-10 test data (label = 0, normal)
            for img, _ in cifar_dataset:
                self.cifar_data.append(img)
                self.labels.append(0)  # Normal

            # Add MNIST test data (label = 1, anomaly)
            for img, _ in mnist_dataset:
                self.mnist_data.append(img)
                self.labels.append(1)  # Anomaly

            self.all_data = self.cifar_data + self.mnist_data

        def __len__(self):
            return len(self.all_data)

        def __getitem__(self, idx):
            return self.all_data[idx], self.labels[idx]

    # Create combined test dataset
    test_dataset = AnomalyTestDataset(cifar_test, mnist_test)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


def encode_dataset_anomaly(encoder, data_loader, device='cuda'):
    encoder.eval()
    encoder.to(device)
    all_features = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            features = encoder(images)

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_images.append(images.cpu())

            if (batch_idx + 1) % 50 == 0:
                print(f"   Processed {batch_idx + 1} batches...")

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)
    all_images = torch.cat(all_images, dim=0)

    return all_features, all_labels, all_images


def calculate_inverse_knn_scores(train_features, test_features, k=2):
    # Fit k-NN on training data
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nbrs.fit(train_features)

    # Find k nearest neighbors for each test sample
    distances, indices = nbrs.kneighbors(test_features)

    # Calculate inverse density score: average distance to k nearest neighbors
    inverse_scores = np.mean(distances, axis=1)

    return inverse_scores


def plot_roc_curve_anomaly(test_labels, scores, model_name, save_path):
    """
    Plot ROC curve with AUC score for anomaly detection
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels, scores)
    roc_auc = auc(fpr, tpr)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}\nAnomaly Detection (MNIST as anomaly)',
              fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return roc_auc


def plot_highest_scoring_anomalies(scores, images, labels, model_name, top_k=7):
    """
    Plot the top_k images with highest anomaly scores
    """
    # Get indices of top scoring images
    top_indices = np.argsort(scores)[-top_k:][::-1]

    # Create plot
    fig, axes = plt.subplots(1, top_k, figsize=(16, 3))
    fig.suptitle(f'Top {top_k} Highest Anomaly Scores - {model_name}',
                 fontsize=14, y=1.05)

    for i, idx in enumerate(top_indices):
        img = images[idx]
        score = scores[idx]
        label = labels[idx]

        # Denormalize and plot
        img_np = denormalize_cifar(img)
        axes[i].imshow(img_np)

        # Create title with score and data type
        data_type = "MNIST" if label == 1 else "CIFAR-10"
        axes[i].set_title(f'Score: {score:.3f}\n({data_type})',
                          fontsize=10)
        axes[i].axis('off')

        # Add colored border (red for anomaly, blue for normal)
        border_color = 'red' if label == 1 else 'blue'
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)

    plt.tight_layout()

    # Save plot
    os.makedirs('./results', exist_ok=True)
    safe_model_name = model_name.replace(" ", "_").lower()
    filename = f'./results/top_anomalies_{safe_model_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def run_anomaly_detection():
    """
    Main function to run complete anomaly detection analysis
    """

    # Load data
    train_loader, test_loader = load_anomaly_data()

    # Define models to analyze
    models = {
        'Regular VICReg': './results/encoder_30.pth',
        'VICReg with Neighbors': './results/encoder_abl2_30.pth'
    }

    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run analysis for each model
    for model_name, model_path in models.items():
        encoder = Encoder()
        if device == 'cpu':
            encoder.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            encoder.load_state_dict(torch.load(model_path))

        print(f"✅ Loaded model: {model_name}")

        # Encode training data (normal data)
        train_features, train_labels, train_images = encode_dataset_anomaly(
            encoder, train_loader, device
        )

        # Encode test data (normal + anomaly)
        test_features, test_labels, test_images = encode_dataset_anomaly(
            encoder, test_loader, device
        )

        # Calculate inverse k-NN scores
        anomaly_scores = calculate_inverse_knn_scores(
            train_features, test_features, k=2
        )

        # Plot ROC curve
        os.makedirs('./results', exist_ok=True)
        safe_model_name = model_name.replace(" ", "_").lower()
        roc_path = f'./results/roc_curve_{safe_model_name}.png'

        auc_score = plot_roc_curve_anomaly(
            test_labels, anomaly_scores, model_name, roc_path
        )

        # Plot highest scoring images
        plot_highest_scoring_anomalies(
            anomaly_scores, test_images, test_labels, model_name, top_k=7
        )

        results[model_name] = {
            'auc_score': auc_score,
            'anomaly_scores': anomaly_scores,
            'test_labels': test_labels,
            'test_images': test_images
        }

    return results


# ==================== CLUSTERING ANALYSIS FUNCTIONS ====================

def encode_cifar10_training(encoder, device='cuda'):

    # Load CIFAR-10 training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0)

    encoder.eval()
    encoder.to(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            features = encoder(images)

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"   Processed {batch_idx + 1} batches...")

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)
    return features, labels


def perform_kmeans_clustering(features, n_clusters=10, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    cluster_centers = kmeans.cluster_centers_
    return cluster_labels, cluster_centers, kmeans


def perform_tsne_reduction(features, perplexity=30, random_state=42):
    if len(features) > 10000:
        indices = np.random.choice(len(features), 10000, replace=False)
        features_subset = features[indices]
    else:
        features_subset = features
        indices = np.arange(len(features))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state,
                n_iter=1000, verbose=0)
    features_2d = tsne.fit_transform(features_subset)
    return features_2d, indices


def plot_clustering_results(features_2d, cluster_labels, true_labels, cluster_centers_2d,
                            method_name, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Colored by cluster labels
    scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1],
                           c=cluster_labels, cmap='tab10', alpha=0.6, s=20)

    # Plot cluster centers
    ax1.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1],
                c='black', marker='x', s=200, linewidths=3, label='Cluster Centers')

    ax1.set_title(f'{method_name}\nColored by Cluster Index', fontsize=12)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Colored by true class labels
    scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1],
                           c=true_labels, cmap='tab10', alpha=0.6, s=20)

    # Plot cluster centers
    ax2.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1],
                c='black', marker='x', s=200, linewidths=3, label='Cluster Centers')

    ax2.set_title(f'{method_name}\nColored by True Class Index', fontsize=12)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def calculate_clustering_metrics(features, cluster_labels, true_labels):
    silhouette = silhouette_score(features, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)

    return silhouette, ari


def run_clustering_analysis():
    # Check available models
    models = {
        'Regular VICReg': './results/encoder_30.pth',
        'VICReg with Neighbors': './results/encoder_abl2_30.pth'
    }

    available_models = {}
    for name, path in models.items():
        if os.path.exists(path):
            available_models[name] = path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    # Process each model
    for model_name, model_path in available_models.items():
        print(f"\n🔍 Processing {model_name}...")
        print("=" * 50)

        # Load encoder
        try:
            encoder = Encoder()
            if device == 'cpu':
                encoder.load_state_dict(torch.load(model_path, map_location='cpu'))
            else:
                encoder.load_state_dict(torch.load(model_path))
            print(f"✅ Loaded {model_name}")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            continue

        # Q1: Encode features and perform K-means clustering
        features, true_labels = encode_cifar10_training(encoder, device)
        cluster_labels, cluster_centers, kmeans = perform_kmeans_clustering(features, n_clusters=10)

        # Q2: Perform t-SNE and visualize
        features_2d, indices = perform_tsne_reduction(features)
        cluster_labels_subset = cluster_labels[indices]
        true_labels_subset = true_labels[indices]
        cluster_centers_2d = []
        for i in range(10):
            cluster_mask = cluster_labels_subset == i
            if np.any(cluster_mask):
                center_2d = np.mean(features_2d[cluster_mask], axis=0)
                cluster_centers_2d.append(center_2d)
            else:
                cluster_centers_2d.append([0, 0])  # Default if no points in cluster
        cluster_centers_2d = np.array(cluster_centers_2d)

        # Create visualization
        safe_name = model_name.replace(' ', '_').lower()
        save_path = f'./results/clustering_{safe_name}.png'
        plot_clustering_results(features_2d, cluster_labels_subset, true_labels_subset,
                                cluster_centers_2d, model_name, save_path)

        # Q3: Calculate metrics
        silhouette, ari = calculate_clustering_metrics(features, cluster_labels, true_labels)
        results[model_name] = {
            'features': features,
            'cluster_labels': cluster_labels,
            'true_labels': true_labels,
            'silhouette_score': silhouette,
            'adjusted_rand_score': ari,
            'cluster_centers': cluster_centers
        }

        model_names = list(results.keys())
        model1, model2 = model_names[0], model_names[1]

        sil1 = results[model1]['silhouette_score']
        sil2 = results[model2]['silhouette_score']

        print(f"{model1}:")
        print(f"   Silhouette Score: {sil1:.4f}")
        print(f"\n{model2}:")
        print(f"   Silhouette Score: {sil2:.4f}")
    return results


def show_menu():
    """Display menu options"""
    print("1. Train Standard VICReg model (complete)")
    print("2. Train VICReg model (μ=0 ablation)")
    print("3. Visualize trained model (PCA/t-SNE)")
    print("4. Train classifier (choose from available models)")
    print("5. Train VICReg with nearest neighbors (ablation 2)")
    print("6. Sample and compare nearest/farthest neighbors")
    print("7. Run anomaly detection analysis (Inverse k-NN)")
    print("8. Debug VICReg training quality")
    print("9. Run clustering analysis (BONUS)")



def run_regular_training(train_loader, test_loader):
    """Train the standard VICReg model (complete implementation)"""
    print("🚀 Starting Standard VICReg Training")

    # Initialize models
    encoder = Encoder()
    projector = Projector(D=128)
    # Train the model with all loss terms
    inv, var, cov, test = train_VICReg_regular(
        encoder=encoder,
        projector=projector,
        train_loader=train_loader,
        test_loader=test_loader
    )

    # Plot training curves   Higher values indicate better-defined clusters")
    #         print(f"   Adjusted Rand Index measures agreement with true class labels")
    #         print(f"   Higher values indicate better class separation")
    #
    #         print(f"\n💡 Visual Analysis Tips:")
    #         print(f"   - Look for tight, well-separated clusters in the left plots")
    #         print(f"   - Look for color consistency within clusters in the right plots")
    #         print(f"   - Black X marks show cluster centers")
    #         print(f"   - Better methods should show clear cluster boundaries")
    #
    #     print(f"\n✅ Clustering analysis complete!")
    #     print(f"📁 All results saved to ./results/ directory")
    plot_regular_losses(inv, var, cov, test, num_epochs=30)


def run_training(train_loader, test_loader):
    print(" Starting VICReg Ablation Study (μ=0)")

    # Initialize models
    encoder = Encoder()
    projector = Projector(D=128)

    inv, var, cov, test = train_VICReg_ablation(
        encoder=encoder,
        projector=projector,
        train_loader=train_loader,
        test_loader=test_loader,
        mu=0  # ABLATION: No variance term
    )

    # Plot training curves
    plot_losses(inv, var, cov, test, num_epochs=30)


def run_visualization(test_loader):
    pca_2d, tsne_2d, colors = two_d_reduction(
        test_loader=test_loader,
        model_path='./results/encoder_abl1_30.pth'
    )
    plot_2d(pca_2d, tsne_2d, colors)


def show_classifier_menu():
    """Display classifier model selection menu"""
    print("Choose Encoder Model for Classification:")
    print("1.  Regular VICReg (encoder_30.pth)")
    print("2.VICReg Ablation μ=0 (encoder_abl1_30.pth)")
    print("3. VICReg with Neighbors (encoder_abl2_30.pth)")
    print("4. Compare existing classifier results")
    print("5. Back to main menu")


def run_classifier_training():
    """Train classifier on frozen features with model selection"""
    # In your train_classifier function, BEFORE the training loop:
    normalized_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_dataset_norm = CIFAR10(root='./data', train=True, transform=normalized_transform)
    test_dataset_norm = CIFAR10(root='./data', train=False, transform=normalized_transform)

    train_loader = DataLoader(train_dataset_norm, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset_norm, batch_size=256, shuffle=False, num_workers=0)

    # Define available models
    models = {
        '1': {
            'name': 'Regular VICReg',
            'path': './results/encoder_30.pth',
            'description': 'Standard VICReg with all loss components'
        },
        '2': {
            'name': 'VICReg Ablation μ=0',
            'path': './results/encoder_abl1_30.pth',
            'description': 'VICReg without variance term'
        },
        '3': {
            'name': 'VICReg with Neighbors',
            'path': './results/encoder_abl2_30.pth',
            'description': 'VICReg using nearest neighbors instead of augmentations'
        }
    }

    # Check which models are available
    available_models = {}
    for key, model_info in models.items():
        if os.path.exists(model_info['path']):
            available_models[key] = model_info

    if not available_models:
        print("❌ No trained models found! Please train models first.")
        print("💡 Available training options:")
        print("   - Option 1: Train Standard VICReg")
        print("   - Option 2: Train VICReg Ablation (μ=0)")
        print("   - Option 5: Train VICReg with Neighbors")
        return None

    # Show available models
    print(f"\n📊 Found {len(available_models)} trained model(s):")
    for key, model_info in available_models.items():
        print(f"   {key}. ✅ {model_info['name']}")

    # Show menu and get user choice
    while True:
        show_classifier_menu()
        choice = input("👉 Select option (1-5): ").strip()

        if choice == '5':
            print("🔙 Returning to main menu...")
            return None

        if choice == '4':
            print("Feature not implemented yet")
            continue

        if choice in available_models:
            selected_model = available_models[choice]
            # Confirm selection
            confirm = input("🤔 Proceed with this model? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes']:
                continue

            # Train classifier
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            classifier, final_accuracy = train_classifier(
                train_loader, test_loader,
                device=device,
                num_epochs=30,
                model_path=selected_model['path']
            )

            model_suffix = selected_model['name'].replace(' ', '_').replace('μ', 'mu').lower()
            classifier_path = f'./results/classifier_{model_suffix}.pth'
            torch.save(classifier.state_dict(), classifier_path)

            return classifier

        else:
            print("\n🔙 Returning to main menu...")
            return None




def main():
    """Main function with interactive menu"""
    # Check for CUDA availability
    print(f"🚀 PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(
            f"🚀 Training on GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB)")
    else:
        print("🚀 Training on CPU")

    # Create results directory
    os.makedirs('./results', exist_ok=True)

    train_loader, test_loader = load_data()
    while True:
        show_menu()
        choice = input("\n👉 Enter your choice (1-11): ").strip()

        if choice == '1':
            run_regular_training(train_loader, test_loader)

        elif choice == '2':
            run_training(train_loader, test_loader)

        elif choice == '3':
            run_visualization(test_loader)

        elif choice == '4':
            run_classifier_training()

        elif choice == '5':
            run_ablation2_training(train_loader, test_loader)

        elif choice == '6':
            sample_and_compare_models(train_loader)

        elif choice == '7':
            run_anomaly_detection()

        elif choice == '8':
            print("Debug function not implemented yet")

        elif choice == '9':
            run_clustering_analysis()



if __name__ == "__main__":
    # Note: num_workers=0 used throughout for Windows compatibility
    main()