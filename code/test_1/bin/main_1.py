import torch
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


#########################
class Encoder(nn.Module):
    def __init__(self, latent_dim=1):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(

            ####block one#############
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            ####block two#############
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            ####block three#############
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, latent_dim=1):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(

            # block three
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # block two
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # block one
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def denormalize(x):
    # Since images are already in the range [0, 1], return them as is
    return x


def save_decoded_images(epoch, original, decoded, save_dir='./saved_images', n=10):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, n, figsize=(20, 4))
    for i in range(n):
        # Apply denormalization
        original_img = denormalize(original[i])
        decoded_img = denormalize(decoded[i])

        # Convert tensors to numpy arrays for plotting
        original_img = original_img.numpy().transpose(1, 2, 0)
        decoded_img = decoded_img.numpy().transpose(1, 2, 0)

        # Display original images
        ax = axes[0, i]
        ax.imshow(original_img)
        ax.set_title('Original')
        ax.axis('off')

        # Display reconstructed images
        ax = axes[1, i]
        ax.imshow(decoded_img)
        ax.set_title('Reconstructed')
        ax.axis('off')

    # Save the figure
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_reconstructions.png'))
    plt.close()




#############################
####implementation###########
if __name__ == "__main__":

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('start training process now')
    print('GPU available:', torch.cuda.is_available())
    print('GPU type:', torch.cuda.get_device_name(0))

    # Simple transformation to scale images to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts image to PyTorch tensor and scales to [0, 1]
    ])

    # Download and load the training data
    trainset = datasets.CIFAR10(root='../../data/CIFAR-10/', download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # Assume `trainset` and `trainloader` are already defined from the training section.
    valset = datasets.CIFAR10(root='../../data/CIFAR-10/', download=True, train=False, transform=transform)
    valloader = DataLoader(valset, batch_size=10, shuffle=False)  # Using small batch for visualization

    # Instantiate the model
    latent_dim = 64  # Adjusted for better capacity in handling CIFAR-10
    autoencoder = Autoencoder(latent_dim=latent_dim).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)

    # Number of epochs
    epochs = 30

    if os.path.exists('./saved_images'):
        shutil.rmtree('./saved_images', ignore_errors=False, onerror=None)

    # Training loop
    for epoch in range(epochs):
        print('current epoch', epoch)
        for images, _ in trainloader:
            images = images.to(device)
            
            # Forward pass
            outputs = autoencoder(images)
            loss = criterion(outputs, images)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation and visualization
        with torch.no_grad():
            for val_images, _ in valloader:
                val_images = val_images.to(device)
                decoded_imgs = autoencoder(val_images)
                save_decoded_images(epoch + 1, val_images.cpu(), decoded_imgs.cpu(), './saved_images')
                break  # Only do this for the first batch
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print('Training completed')







