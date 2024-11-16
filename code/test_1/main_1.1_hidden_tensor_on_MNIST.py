import os
import torch
import numpy as np
import shutil
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


###############################################
#### Implement pytorch autoencoder on MNIST ###

# Define the encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 32, 14, 14]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 64, 7, 7]
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


# Define the decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch_size, 64, 14, 14]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch_size, 32, 28, 28]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),  # [batch_size, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.decoder(z)
        return x


# Combine encoder and decoder into an autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed


# Function to combine original and generated images into a single grid
def save_combined_images(model, data_loader, save_path, num_images=10):
    model.eval()

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            reconstructed = model(data)
            break  # Only need the first batch

    # Move tensors to CPU and convert to numpy arrays
    data = data.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    # Create a figure with subplots
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

    for i in range(num_images):
        # Plot original images in the first row
        axes[0, i].imshow(data[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')

        # Plot reconstructed images in the second row
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')

    # Adjust layout
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Save the combined figure
    plt.savefig(save_path + 'combined_reconstruction.png')
    plt.close()

    print(f"Combined image saved to '{save_path}'.")




#############################
####implementation###########
if __name__ == "__main__":

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 20
    save_path = "../../model_bin/main_1.1_Nov_15_2024/"


    # Data preprocessing and loading
    transform = transforms.Compose([
        transforms.ToTensor(),  # This will convert the data to range [0, 1]
        # Remove the normalization for BCELoss
    ])

    train_dataset = datasets.MNIST(root='../../data/MNIST/', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='../../data/MNIST/', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    # Instantiate the model, define the loss function and the optimizer
    model = Autoencoder().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            # Forward pass
            reconstructed = model(data)
            loss = criterion(reconstructed, data)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Remove the directory if it already exists
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    # Save the model
    torch.save(model.state_dict(), save_path + 'autoencoder.pth')

    # Visualize
    save_combined_images(model, test_loader, './saved_image_MNIST_hidden_tensor/')















