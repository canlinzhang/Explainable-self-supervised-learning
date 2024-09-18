import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import shutil

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Assuming input is grayscale; if not, change 1 to 3 in the first Conv2d layer.
        self.encoder = nn.Sequential(
            
            ######block one##############
            nn.Conv2d(3, 64, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),

            ######block two##############
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),

            ######block three##############
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
                        
            ######block four##############
            nn.Conv2d(256, 512, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),

            ######block five##############
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  # Changed from 1 to 3 for CIFAR-10 RGB channels
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        
        self.decoder = nn.Sequential(

            ######block five##############
            nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1),
            
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  
            
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  
            
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  
            
            ######block four##############
            nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1),
            
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  
            
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),  
            
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),  
            
            ######block three##############
            nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
            
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            
            
            #####block two#################
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),  
            
            
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),             
            
            #####block one#################
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),  
            
            
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Sigmoid()  # Ensure the final output is between 0 and 1
        )

    def forward(self, x):
        #print("Input:", x.shape)
        #print('    ')
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            #print(f"Encoder Layer {i+1}:", x.shape)
            
        #print('    ')
            
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            #print(f"Decoder Layer {i+1}:", x.shape)
        return x


class UnlabeledDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.folder_path, self.image_files[index])
        image = Image.open(img_path).convert('RGB')  # Convert grayscale images to RGB if necessary
        if self.transform:
            image = self.transform(image)
        return image
    

def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)  # Added tqdm here

    count = 0
    for inputs in progress_bar:
        inputs = inputs.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Update the progress bar with the current loss
        progress_bar.set_postfix(loss=loss.item())

        count += 1
        #  !!!!! comment this out in actual practice
        #if count == 200:
        #    break
        
    return running_loss / len(dataloader.dataset)


def train_epoch_center_32(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)  # Added tqdm here

    count = 0
    for inputs in progress_bar:
        inputs = inputs.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Extract the middle 32x32 region from the inputs and outputs
        center_region = lambda x: x[:, :, 112:144, 112:144]  # Assuming inputs are 256x256
        outputs_center = center_region(outputs)
        inputs_center = center_region(inputs)
        
        # Calculate loss only on the 32x32 center region
        loss = criterion(outputs_center, inputs_center)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs_center.size(0)
        
        # Update the progress bar with the current loss
        progress_bar.set_postfix(loss=loss.item())

        count += 1
        #  !!!!! comment this out in actual practice
        #if count == 200:  # Early stopping for testing
        #    break
        
    return running_loss / len(dataloader.dataset)


def validate_epoch(model, dataloader, criterion, device, epoch, save_images=True, output_dir='./output_images/'):
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)  # Added tqdm here
    
    # Ensure the output directory exists
    if save_images:
        if os.path.exists(output_dir) and epoch == 0:
            shutil.rmtree(output_dir, ignore_errors=False, onerror=None)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    with torch.no_grad():
        for batch_idx, inputs in enumerate(progress_bar):
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_loss += loss.item() * inputs.size(0)
            
            # Update the progress bar with the current loss
            progress_bar.set_postfix(loss=loss.item())

            # Save output images
            if save_images and batch_idx % 50 == 0:  # Save images from every 50th batch, adjust as needed
                image_filename = f'epoch_{epoch}_output_{batch_idx}.png'
                save_image(outputs, os.path.join(output_dir, image_filename))

            # !!!!! comment this out in actual practice
            #if batch_idx == 200:
            #    break
            
    return running_loss / len(dataloader.dataset)


def validate_epoch_center_32(model, dataloader, criterion, device, epoch, save_images=True, output_dir='./output_images/'):
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)  # Added tqdm here
    
    # Ensure the output directory exists
    if save_images:
        if os.path.exists(output_dir) and epoch == 0:
            shutil.rmtree(output_dir, ignore_errors=False, onerror=None)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    with torch.no_grad():
        for batch_idx, inputs in enumerate(progress_bar):
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Extract the middle 32x32 region from the inputs and outputs
            center_region = lambda x: x[:, :, 112:144, 112:144]  # Assuming inputs are 256x256
            outputs_center = center_region(outputs)
            inputs_center = center_region(inputs)
            
            # Calculate loss only on the 32x32 center region
            loss = criterion(outputs_center, inputs_center)
            running_loss += loss.item() * inputs_center.size(0)
            
            # Update the progress bar with the current loss
            progress_bar.set_postfix(loss=loss.item())

            # Save output images
            if save_images and batch_idx % 50 == 0:  # Save images from every 50th batch, adjust as needed
                image_filename = f'epoch_{epoch}_output_{batch_idx}.png'
                save_image(outputs, os.path.join(output_dir, image_filename))  # Ensure to handle RGB normalization if needed

            # !!!!! comment this out in actual practice.
            #if batch_idx == 200:  # Early stopping for testing
            #    break
            
    return running_loss / len(dataloader.dataset)





#################################
####implementation###############
if __name__ == "__main__":

    base_dir = '../model_bin/main_test_1_Aug_30_2024/'

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=False, onerror=None)
    os.makedirs(base_dir)
    
    print('start training process now')
    print('GPU available:', torch.cuda.is_available())
    print('GPU type:', torch.cuda.get_device_name(0))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the datasets using the custom UnlabeledDataset
    train_dataset = UnlabeledDataset(folder_path='../data/train_combo/', transform=transform)
    val_dataset = UnlabeledDataset(folder_path='../data/val_combo/', transform=transform)

    # Define the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)


    # Define the model, criterion and optimizer
    autoencoder = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.encoder.parameters(), lr=0.001)

    # Training the model
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_epoch(autoencoder, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(autoencoder, val_loader, criterion, device, epoch, 
                                  save_images=True, output_dir='../results/main_test_1_Aug_30_2024/')
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Save model snapshots
        if epoch % 2 == 1:
            model_path = os.path.join(base_dir, f'autoencoder_epoch_{epoch+1}.pth')
            torch.save(autoencoder.state_dict(), model_path)

    



