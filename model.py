import math
import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random

# -----------------------------------------------------------------------------
#  Configuration
# -----------------------------------------------------------------------------
BATCH_SIZE = 64
LATENT_DIM = 200
NUM_EPOCHS = 30
"""
the boolean global variables are used to control the flow of the program
each variable corresponds to a specific question or task
"""
TRAIN_AMOR = False
TRAIN_LATENT = False
Q1 = False
Q2 = False
Q3 = False
Q4 = False


# -----------------------------------------------------------------------------
#  Utility: wrap a Subset so __getitem__ returns (x, y, idx)
# -----------------------------------------------------------------------------
class IndexedSubset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, y, idx


# -----------------------------------------------------------------------------
#  Simple "hash" for an image (must be consistent train ↔ sample)
# -----------------------------------------------------------------------------
def get_image_id(img_tensor):
    return float(torch.sum(img_tensor).item())


# -----------------------------------------------------------------------------
#  MNIST loaders: one plain, one that also returns idx
# -----------------------------------------------------------------------------
def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Full MNIST, then pick a 20k-example stratified subset
    full_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    targets = full_train.targets
    train_idx, _ = train_test_split(
        list(range(len(targets))),
        train_size=20000,
        stratify=targets,
        random_state=0
    )
    train_subset = Subset(full_train, train_idx)
    indexed_train_ds = IndexedSubset(train_subset)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_idx = DataLoader(indexed_train_ds, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, train_loader_idx, test_loader, train_subset


# -----------------------------------------------------------------------------
#  Model definition with fc_decode
# -----------------------------------------------------------------------------
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28→14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 14→7
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 7→4
            nn.ReLU(),
            nn.Conv2d(128, 128, 2)  # 4→3→2→1
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        # <— renamed back to fc_decode
        self.fc_decode = nn.Linear(latent_dim, 128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2),  # 1→2
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, output_padding=1),  # 2→4
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1),  # 4→7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),  # 7→14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, output_padding=1),  # 14→28
        )

    def encode(self, x):
        h = self.encoder(x)
        h = F.adaptive_avg_pool2d(h, 1).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z).view(-1, 128, 1, 1)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# -----------------------------------------------------------------------------
#  Amortized training (Q1)
# -----------------------------------------------------------------------------
def training_amortized(train_loader, test_loader, model, loss_fn, num_epochs=NUM_EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            recon_loss = loss_fn(recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Amortized] Epoch {epoch + 1}/{num_epochs}  Train loss: {avg_loss:.4f}")

        # ── Save a checkpoint every 5 epochs ──
        if (epoch + 1) % 5 == 0:
            ckpt_path = f"model_Amortized_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ checkpoint saved at {ckpt_path}")

    # final save at end of training
    final_path = f"model_Amortized_epoch_{num_epochs}.pth"
    torch.save(model.state_dict(), final_path)
    print(f"✅ Amortized model saved as {final_path}")


# -----------------------------------------------------------------------------
#  Latent‐vector optimization (Q3) - with improved indexing
# -----------------------------------------------------------------------------
def training_latent_opt(train_loader_idx, train_subset, model, loss_fn, mu_q, r_q, num_epochs=NUM_EPOCHS):
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-3},
        {'params': model.fc_decode.parameters(), 'lr': 1e-3},
        {'params': [mu_q, r_q], 'lr': 1e-2},
    ])

    # Create a mapping dictionary - enhanced to store both image ID and label
    image_mapping = {}

    # Create separate image databases by digit class
    digit_to_indices = {d: [] for d in range(10)}

    # Pre-process the entire dataset to create mappings
    print("Creating image mappings...")
    for batch_idx, (x_batch, y_batch, idx_batch) in enumerate(train_loader_idx):
        for img, label, idx in zip(x_batch, y_batch, idx_batch):
            img_id = get_image_id(img)
            image_mapping[img_id] = {
                'index': int(idx),
                'label': int(label)
            }
            digit_to_indices[int(label)].append(int(idx))

        if batch_idx % 10 == 0:
            print(f"Processed {batch_idx} batches...")

    print(f"Image mapping created with {len(image_mapping)} entries")
    for d in range(10):
        print(f"Class {d}: {len(digit_to_indices[d])} examples")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        for x, y, idx in train_loader_idx:
            mu_b = mu_q[idx]
            sigma_b = torch.exp(0.5 * r_q[idx])
            z = mu_b + sigma_b * torch.randn_like(sigma_b)

            recon = model.decode(z)
            recon_loss = loss_fn(recon, x)
            kl = -0.5 * torch.mean(1 + r_q[idx] - mu_b.pow(2) - torch.exp(r_q[idx]))
            loss = recon_loss + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg = running_loss / len(train_loader_idx)
        print(f"[Latent] Epoch {epoch + 1}/{num_epochs}  Train loss: {avg:.4f}")

        # checkpoint
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'mu_q': mu_q.detach().cpu(),
                'r_q': r_q.detach().cpu(),
                'image_mapping': image_mapping,
                'digit_to_indices': digit_to_indices
            }
            path = f"model_latent_Optimization_epoch_{epoch}.pth"
            torch.save(ckpt, path)
            print(f"  ↳ saved: {path} with {len(image_mapping)} image mappings")

    print("✅ Latent‐optimized model saved.")


# -----------------------------------------------------------------------------
#  Sampling Q1 (amortized)
# -----------------------------------------------------------------------------
def sample_images_amortized(epochs=[1, 5, 10, 20, 30], device='cpu', split='train'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = torchvision.datasets.MNIST(root='./data', train=(split == 'train'), download=True, transform=transform)

    # pick one of each digit
    found, samples = set(), []
    for img, lbl in ds:
        if lbl not in found:
            samples.append((img, lbl))
            found.add(lbl)
        if len(found) == 10:
            break

    all_recons = []
    for e in epochs:
        model = ConvVAE().to(device)
        model.load_state_dict(torch.load(f"model_Amortized_epoch_{e}.pth", map_location=device))
        model.eval()

        recons = []
        with torch.no_grad():
            for img, _ in samples:
                img = img.unsqueeze(0).to(device)
                mu, _ = model.encode(img)
                recon = model.decode(mu)
                recons.append(recon.squeeze().cpu())
        all_recons.append(recons)

    # plot
    rows = len(epochs) + 1
    fig, axs = plt.subplots(rows, 10, figsize=(15, 1.5 * rows))
    for c, (img, lbl) in enumerate(samples):
        axs[0, c].imshow(img.squeeze(), cmap='gray')
        axs[0, c].set_title(str(lbl))
        axs[0, c].axis('off')
    axs[0, 0].set_ylabel("orig", rotation=0, labelpad=20)

    for r, e in enumerate(epochs):
        for c in range(10):
            axs[r + 1, c].imshow(all_recons[r][c], cmap='gray')
            axs[r + 1, c].axis('off')
        axs[r + 1, 0].set_ylabel(f"ep {e}", rotation=0, labelpad=20)

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
#  Sampling Q3 (latent) - Fixed to handle int labels correctly
# -----------------------------------------------------------------------------
def sample_images_latent(train_subset, epochs=[0, 5, 10, 20, 29], device='cpu'):
    # Use the first checkpoint to get mappings
    base_ckpt = torch.load(f"model_latent_Optimization_epoch_{epochs[0]}.pth", map_location=device)

    # Get the improved mappings
    image_mapping = base_ckpt.get('image_mapping', {})
    digit_to_indices = base_ckpt.get('digit_to_indices', {})

    if not image_mapping or not digit_to_indices:
        print("❌ Missing mapping information in checkpoint")
        return

    print(f"Loaded image mapping with {len(image_mapping)} entries")

    # Randomly select one image from each digit class
    sample_indices = []
    sample_images = []
    sample_labels = []

    # Get one random image from each digit class
    for digit in range(10):
        if digit in digit_to_indices and digit_to_indices[digit]:
            # Randomly select an index for this digit
            idx = random.choice(digit_to_indices[digit])
            sample_indices.append(idx)

            # Get the actual image from the training subset
            image, label = train_subset[idx]
            sample_images.append(image)
            sample_labels.append(label)
            print(f"Selected index {idx} for digit {digit}")
        else:
            print(f"⚠ No indices found for digit {digit}")

    if len(sample_indices) < 10:
        print(f"⚠ Only found {len(sample_indices)} out of 10 digit samples")

    # Process each epoch
    all_recons = []
    for e in epochs:
        print(f"Processing epoch {e}...")
        ckpt = torch.load(f"model_latent_Optimization_epoch_{e}.pth", map_location=device)
        model = ConvVAE().to(device)
        model.load_state_dict(ckpt['model_state'])
        mu_q = ckpt['mu_q'].to(device)
        r_q = ckpt['r_q'].to(device)
        model.eval()

        recons = []
        with torch.no_grad():
            for idx in sample_indices:
                mu = mu_q[idx].unsqueeze(0)
                r = r_q[idx].unsqueeze(0)
                z = mu + torch.exp(0.5 * r) * torch.randn_like(r)
                rec = model.decode(z)
                recons.append(rec.squeeze().cpu())
        all_recons.append(recons)

    # Plot the results
    rows = len(epochs) + 1
    fig, axs = plt.subplots(rows, len(sample_indices), figsize=(15, 1.5 * rows))

    # Plot original images - FIX: removed .item() since labels are already integers
    for c, (img, lbl) in enumerate(zip(sample_images, sample_labels)):
        axs[0, c].imshow(img.squeeze(), cmap='gray')
        axs[0, c].set_title(str(lbl))  # Fixed: just convert to string, no .item()
        axs[0, c].axis('off')
    axs[0, 0].set_ylabel("orig", rotation=0, labelpad=20)

    # Plot reconstructions
    for r, e in enumerate(epochs):
        for c in range(len(sample_indices)):
            if c < len(all_recons[r]):  # Make sure we don't exceed array bounds
                axs[r + 1, c].imshow(all_recons[r][c], cmap='gray')
                axs[r + 1, c].axis('off')
        axs[r + 1, 0].set_ylabel(f"ep {e}", rotation=0, labelpad=20)

    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------

#  sampling from prior
#-----------------------------------------------------------------------------
def sample_from_prior_amortized(epochs=[1, 5, 15, 20, 30], n_samples=10, device='cpu'):
    # Setup the plot
    fig, axs = plt.subplots(len(epochs), n_samples, figsize=(15, 2 * len(epochs)))
    fig.suptitle('Samples from prior p(z) - Amortized Model', fontsize=16)

    # Sample from prior and generate images using amortized model
    for r, e in enumerate(epochs):
        # Load the amortized model
        try:
            model = ConvVAE().to(device)
            model.load_state_dict(torch.load(f"model_Amortized_epoch_{e}.pth", map_location=device))
            model.eval()

            # Sample from standard normal prior
            with torch.no_grad():
                z = torch.randn(n_samples, LATENT_DIM).to(device)
                # Decode the samples
                samples = model.decode(z)

            # Plot
            for c in range(n_samples):
                img = samples[c].squeeze().cpu()
                axs[r, c].imshow(img, cmap='gray')
                axs[r, c].axis('off')
            axs[r, 0].set_ylabel(f"Epoch {e}", rotation=90, labelpad=10)
        except FileNotFoundError:
            print(f"Model file for amortized model at epoch {e} not found. Skipping.")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    plt.show()
def sample_from_prior_latent(epochs=[0, 5, 10, 20, 29],n_samples=10,device='cpu'):
    # figure with one row per epoch, one column per sample
    fig, axs = plt.subplots(len(epochs), n_samples, figsize=(n_samples, 1.5*len(epochs)))
    fig.suptitle('Samples from prior p(z) – Latent-Optimized Model', y=0.92)

    for row, e in enumerate(epochs):
        # pick the checkpoint file (round down to nearest multiple of 5 if you saved only every 5 epochs)
        ckpt_epoch = e if os.path.exists(f"model_latent_Optimization_epoch_{e}.pth") else (e//5)*5
        ckpt_path  = f"model_latent_Optimization_epoch_{ckpt_epoch}.pth"

        print(f"→ Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_sd = ckpt['model_state']

        # fix any old "fc_dec" → "fc_decode" mismatch
        fixed_sd = {}
        for k, v in raw_sd.items():
            if k.startswith('fc_dec.') and not k.startswith('fc_decode.'):
                new_k = k.replace('fc_dec.', 'fc_decode.')
                fixed_sd[new_k] = v
            else:
                fixed_sd[k] = v

        # build & load
        model = ConvVAE().to(device)
        model.load_state_dict(fixed_sd)
        model.eval()

        # sample and decode
        with torch.no_grad():
            z = torch.randn(n_samples, LATENT_DIM, device=device)
            samples = model.decode(z).cpu()

        # plot
        for col in range(n_samples):
            axs[row, col].imshow(samples[col].squeeze(), cmap='gray')
            axs[row, col].axis('off')

        axs[row, 0].set_ylabel(f"ep {ckpt_epoch}", rotation=0, labelpad=20)

    plt.tight_layout(rect=[0,0,1,0.9])
    plt.show()
def img_log_liklihood(loader,M=1000):
    digit_counter = [0,0,0,0,0,0,0,0,0,0]
    log_likelihoods= {}
    avg_logs = {}
    for i in range(10):
        log_likelihoods[i] = []
    plotting_imgs = [[],[]]
    #load Amortized model
    model = ConvVAE()
    model.load_state_dict(torch.load("model_Amortized_epoch_30.pth"))
    #sample from train loader
    for batch in loader:
        x, y = batch
        for img, lbl in zip(x, y):
            if digit_counter[lbl.item()] < 5:
                digit_counter[lbl.item()] += 1
                log_px = compute_log_px(model,img,M).item()
                log_likelihoods[lbl.item()].append(log_px)
                if digit_counter[lbl.item()]==1: #first image from class
                   plotting_imgs[0].append(img)
                   plotting_imgs[1].append(log_px)
            for i in range(10):
                if len(log_likelihoods[i]) < 5:
                    continue
            break
    for i in range(10):
        avg_logs[i] = np.mean(log_likelihoods[i])
        print(f"Digit {i}: log p(x) = {avg_logs[i]:.2f}")
    plot_images_with_logs(plotting_imgs)
    print("avg per digit: ", avg_logs)
    print("avg cross all: ", np.mean(list(avg_logs.values())))
    #sample from test loader
def compute_log_px(model,img,M=1000):
    model.eval()
    #compute log liklihood of images
    with torch.no_grad():
        # sample z from  q distribution
        mu, logvar = model.encode(img.unsqueeze(0))
        eps = torch.randn(M, mu.size(-1), device=mu.device)
        z =mu + eps * torch.exp(0.5*logvar)
        log_pz = -0.5 * ((z ** 2) + math.log(2 * math.pi)).sum(dim=1)
        var = torch.exp(logvar)
        log_qzx = -0.5 * (
                ((z - mu) ** 2) / var + torch.log(var) + math.log(2 * math.pi)
        ).sum(dim=1)
        recon = model.decode(z)
        img_rep = img.unsqueeze(0).expand_as(recon)
        log_px_z = -0.5 * ((img_rep - recon)**2 + math.log(2*math.pi)).sum(dim=[1,2,3])
        log_w = log_pz + log_px_z - log_qzx
        return torch.logsumexp(log_w, dim=0) - math.log(M)








                        # -----------------------------------------------------------------------------


def plot_images_with_logs(data_list):
    """
    Plot a row of images each with its corresponding log-probability as title.

    Parameters:
    - data_list: a list/tuple of two lists:
        * data_list[0]: list of image tensors or numpy arrays (length N)
        * data_list[1]: list of floats, log-probabilities (length N)
    """
    images, log_probs = data_list
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(2 * n, 2))

    for i, (img, logp) in enumerate(zip(images, log_probs)):
        ax = axs[i] if n > 1 else axs
        # Convert torch.Tensor to numpy if needed
        if hasattr(img, 'detach'):
            img_np = img.detach().cpu().numpy()
        else:
            img_np = img
        # Remove channel dimension if present
        img_np = img_np.squeeze()

        ax.imshow(img_np, cmap='gray')
        ax.set_title(f"{logp:.2f}", fontsize=16)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

#  Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_loader, train_loader_idx, test_loader, train_subset = load_dataset()

    # Q1: amortized
    if TRAIN_AMOR:
        model, loss_fn = ConvVAE(), nn.MSELoss()
        training_amortized(train_loader, test_loader, model, loss_fn)

    if Q1:
        sample_images_amortized(split='test')
        sample_images_amortized(split='train')

    # Q2: sampling from prior
    if Q2:
        sample_from_prior_amortized(epochs=[1, 5, 10, 20, 30], n_samples=10)
        sample_from_prior_latent(epochs=[1, 5, 10, 20, 30], device='cpu')

    # Q3: latent‐vector optimization + sampling
    if Q3:
        if TRAIN_LATENT:
            model, loss_fn = ConvVAE(), nn.MSELoss()
            N = len(train_loader.dataset)
            mu_q = nn.Parameter(0.01 * torch.randn(N, LATENT_DIM))
            r_q = nn.Parameter(torch.zeros(N, LATENT_DIM))
            training_latent_opt(train_loader_idx, train_subset, model, loss_fn, mu_q, r_q)

        sample_images_latent(train_subset)
    # Q4: computing log likelihood
    if Q4:
        img_log_liklihood(train_loader)
        img_log_liklihood(test_loader)