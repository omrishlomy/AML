"""
Debug Anomaly Detection Issues
==============================

This script helps debug why CIFAR-10 images are getting higher anomaly scores than MNIST.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import os

# ==================== QUICK DEBUG FUNCTIONS ====================

def debug_data_preprocessing():
    """Check if MNIST and CIFAR-10 are being processed correctly"""
    print("🔍 DEBUGGING DATA PREPROCESSING")
    print("=" * 40)

    # CIFAR-10 transform
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # MNIST transform
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Load a few samples
    cifar_test = CIFAR10(root='./data', train=False, download=False, transform=cifar_transform)
    mnist_test = MNIST(root='./data', train=False, download=False, transform=mnist_transform)

    # Get first samples
    cifar_img, _ = cifar_test[0]
    mnist_img, _ = mnist_test[0]

    print(f"CIFAR-10 shape: {cifar_img.shape}")
    print(f"MNIST shape: {mnist_img.shape}")
    print(f"CIFAR-10 range: [{cifar_img.min():.3f}, {cifar_img.max():.3f}]")
    print(f"MNIST range: [{mnist_img.min():.3f}, {mnist_img.max():.3f}]")

    # Visualize samples
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    # Denormalize for visualization
    def denormalize(tensor):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)
        denorm = tensor * std + mean
        return torch.clamp(denorm, 0, 1).permute(1, 2, 0).numpy()

    axes[0].imshow(denormalize(cifar_img))
    axes[0].set_title("CIFAR-10 Sample")
    axes[0].axis('off')

    axes[1].imshow(denormalize(mnist_img))
    axes[1].set_title("MNIST Sample (Processed)")
    axes[1].axis('off')

    # Show more samples
    cifar_img2, _ = cifar_test[1]
    mnist_img2, _ = mnist_test[1]

    axes[2].imshow(denormalize(cifar_img2))
    axes[2].set_title("CIFAR-10 Sample 2")
    axes[2].axis('off')

    axes[3].imshow(denormalize(mnist_img2))
    axes[3].set_title("MNIST Sample 2")
    axes[3].axis('off')

    plt.suptitle("Data Preprocessing Check", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Check if MNIST looks reasonable
    print("\n💡 Do the MNIST images look like proper digits converted to 3-channel RGB?")
    print("💡 They should be recognizable digits but in color format.")

    return cifar_img, mnist_img

def debug_score_distributions():
    """Check the actual score distributions for CIFAR-10 vs MNIST"""
    print("\n🔍 DEBUGGING SCORE DISTRIBUTIONS")
    print("=" * 40)

    # This assumes you've already run the main anomaly detection
    # Let's simulate with a simple check

    # Load a small subset and compute distances manually
    cifar_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Load small subsets
    cifar_train = CIFAR10(root='./data', train=True, download=False, transform=cifar_transform)
    cifar_test = CIFAR10(root='./data', train=False, download=False, transform=cifar_transform)
    mnist_test = MNIST(root='./data', train=False, download=False, transform=mnist_transform)

    # Take small samples for quick test
    n_train = 1000
    n_test = 200

    print(f"Taking {n_train} training samples and {n_test} test samples each...")

    # Get training data (CIFAR-10)
    train_images = torch.stack([cifar_train[i][0] for i in range(n_train)])

    # Get test data
    cifar_test_images = torch.stack([cifar_test[i][0] for i in range(n_test)])
    mnist_test_images = torch.stack([mnist_test[i][0] for i in range(n_test)])

    print(f"Train images shape: {train_images.shape}")
    print(f"CIFAR-10 test shape: {cifar_test_images.shape}")
    print(f"MNIST test shape: {mnist_test_images.shape}")

    # Flatten for simple distance computation
    train_flat = train_images.view(n_train, -1).numpy()
    cifar_test_flat = cifar_test_images.view(n_test, -1).numpy()
    mnist_test_flat = mnist_test_images.view(n_test, -1).numpy()

    print("\nComputing simple pixel-space distances...")

    # Compute k-NN distances in pixel space
    knn = NearestNeighbors(n_neighbors=2, metric='euclidean')
    knn.fit(train_flat)

    # Get distances for CIFAR-10 test
    cifar_distances, _ = knn.kneighbors(cifar_test_flat)
    cifar_scores = np.mean(cifar_distances, axis=1)

    # Get distances for MNIST test
    mnist_distances, _ = knn.kneighbors(mnist_test_flat)
    mnist_scores = np.mean(mnist_distances, axis=1)

    print(f"\nCIFAR-10 test scores: mean={cifar_scores.mean():.3f}, std={cifar_scores.std():.3f}")
    print(f"MNIST test scores: mean={mnist_scores.mean():.3f}, std={mnist_scores.std():.3f}")

    # Plot distributions
    plt.figure(figsize=(10, 6))
    plt.hist(cifar_scores, bins=30, alpha=0.7, label='CIFAR-10 (should be lower)', color='blue')
    plt.hist(mnist_scores, bins=30, alpha=0.7, label='MNIST (should be higher)', color='red')
    plt.xlabel('Inverse k-NN Density Score')
    plt.ylabel('Frequency')
    plt.title('Score Distributions (Pixel Space - Quick Test)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    if mnist_scores.mean() > cifar_scores.mean():
        print("✅ EXPECTED: MNIST scores are higher than CIFAR-10 scores")
    else:
        print("❌ PROBLEM: CIFAR-10 scores are higher than MNIST scores!")
        print("   This suggests an issue with data preprocessing or model")

    return cifar_scores, mnist_scores

def debug_model_embeddings():
    """Check if the VICReg model is producing reasonable embeddings"""
    print("\n🔍 DEBUGGING MODEL EMBEDDINGS")
    print("=" * 40)

    # Check if model exists
    model_path = './results/encoder_30.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    # Import the model (assuming you have the Encoder class available)
    try:
        # You might need to copy the Encoder class definition here
        print("🔧 Loading VICReg model...")
        print(f"Model path: {model_path}")

        # Load model weights
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"✅ Model weights loaded successfully")
        print(f"Model has {len(state_dict)} parameter tensors")

        # Check if weights look reasonable
        first_layer = None
        for name, param in state_dict.items():
            if 'weight' in name:
                first_layer = param
                break

        if first_layer is not None:
            print(f"First layer weight stats:")
            print(f"  Shape: {first_layer.shape}")
            print(f"  Mean: {first_layer.mean():.6f}")
            print(f"  Std: {first_layer.std():.6f}")
            print(f"  Min: {first_layer.min():.6f}")
            print(f"  Max: {first_layer.max():.6f}")

            if first_layer.std() < 1e-6:
                print("⚠️  WARNING: Very low weight variance - model might have collapsed!")
            else:
                print("✅ Weight statistics look reasonable")

    except Exception as e:
        print(f"❌ Error loading model: {e}")

def run_full_debug():
    """Run all debug functions"""
    print("🚨 ANOMALY DETECTION DEBUG SESSION")
    print("=" * 50)

    # Step 1: Check data preprocessing
    cifar_img, mnist_img = debug_data_preprocessing()

    # Step 2: Check score distributions in pixel space
    cifar_scores, mnist_scores = debug_score_distributions()

    # Step 3: Check model
    debug_model_embeddings()

    print("\n💡 DEBUGGING SUMMARY:")
    print("=" * 30)
    print("1. Check if MNIST images look correct in the visualization")
    print("2. Check if MNIST scores are higher than CIFAR-10 scores in pixel space")
    print("3. Check if model weights look reasonable (not collapsed)")
    print("\nIf any of these fail, we've found the issue!")

if __name__ == "__main__":
    run_full_debug()