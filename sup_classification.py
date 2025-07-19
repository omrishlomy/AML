import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

################### Define a CNN architecture ###################
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)         # output: (6, 24, 24)
        self.pool = nn.MaxPool2d(2, 2)            # output: (6, 12, 12) after first pooling
        self.conv2 = nn.Conv2d(6, 16, 5)          # output: (16, 8, 8)
        # Second pooling reduces (16, 8, 8) to (16, 4, 4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)     # Adjusted to match flattened size (256)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # shape: (batch, 6, 12, 12)
        x = self.pool(F.relu(self.conv2(x)))     # shape: (batch, 16, 4, 4)
        x = torch.flatten(x, 1)                  # shape: (batch, 16*4*4 = 256)
        x = F.relu(self.fc1(x))                  # shape: (batch, 120)
        x = F.relu(self.fc2(x))                  # shape: (batch, 84)
        x = self.fc3(x)                          # shape: (batch, 10)
        return x

################### Load MNIST dataset ###################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_targets = train_dataset.targets
train_idx, _ = train_test_split(range(len(train_targets)), train_size=20000, stratify=train_targets)
train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

################### Initialize the model, loss function, and optimizer ###################
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop with additional print statements for tracking progress
val_losses = []
val_accuracies = []
num_epochs = 50

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    batch_count = 0

    # Training iterations with progress bar; we use enumerate to track batch index
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        optimizer.zero_grad()          # Zero the parameter gradients
        outputs = model(images)          # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()                  # Backpropagation to compute gradients
        optimizer.step()                 # Update weights

        batch_loss = loss.item()
        running_loss += batch_loss
        batch_count += 1

        # Print batch-level loss every 100 batches
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], Batch Loss: {batch_loss:.4f}")

    avg_train_loss = running_loss / batch_count
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Validating"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    accuracy = correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy * 100:.2f}%\n")

print("Finished Training")

################### Plot the validation loss and accuracy ###################
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(val_losses, marker='o', label='Validation Loss')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, marker='o', label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
