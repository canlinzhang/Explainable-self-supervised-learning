

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import argparse
import math
import logging

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.utils.data as data

#import utils
#import models.builer as builder
#import dataloader

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms

from tqdm import tqdm
from PIL import Image, ImageFilter


'''Duplicate the https://github.com/Horizon2333/imagenet-autoencoder model on my own.
   I just explicitly code up the entire autoencoder one layer by another.
   Then, implement the same training and preprocess methods.
'''


###############################
####Deep Network###############
###############################
##we mimic VGG model with [2,2,4,4,4]. Is it VGG 16 or VGG19?
#anyway, we each block as 2, 2, 4, 4, 4 conv layers followed by a maxpooling
#Originally, the ImageNet Autoencoder model we have: For the five blockers:
# input_dim=3,   output_dim=64,  hidden_dim=64,  layers=configs[0], enable_bn=True
# input_dim=64,  output_dim=128, hidden_dim=128, layers=configs[1], enable_bn=True
# input_dim=128, output_dim=256, hidden_dim=256, layers=configs[2], enable_bn=True
# input_dim=256, output_dim=512, hidden_dim=512, layers=configs[3], enable_bn=True
# input_dim=512, output_dim=512, hidden_dim=512, layers=configs[4], enable_bn=True

#But now, we want to narrow down the middle hidden layer to build the "small amount of information part"
#We will first reduce cnn filter number. Then, we will add fully connected layers.


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(

            ####block one#############
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ####block two#############
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ####block three#############
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ####block four#############
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            ####block five#############
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            ####block six#############
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            ####linear layers#########
            nn.Flatten(),  # Flatten the output of conv layer
            nn.Linear(16 * 28 * 28, 512),  # Adjust 512 as needed
            nn.Tanh(),
            
            nn.Linear(512, latent_dim),  # Adjust 128 as needed
            nn.Tanh()

        )

    def forward(self, x):
        x = self.encoder(x)
        return x

    #def forward(self, x):
    #    for layer in self.encoder:
    #        x = layer(x)
    #        print(f"Shape after {layer.__class__.__name__}: {x.shape}")
    #    return x
    

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(

            ###linear layers##############
            nn.Linear(latent_dim, 512),
            nn.Tanh(),
            
            nn.Linear(512, 16 * 28 * 28),
            nn.Tanh(),
            
            nn.Unflatten(1, (16, 28, 28)),

            ####block six#################
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

            ####block five#################
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),

            ####block four#################
            #nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2),

            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),

            ####block three#################
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2),

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),

            ####block two#################
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),

            ####block one#################
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    #def forward(self, x):
    #    for layer in self.decoder:
    #        x = layer(x)
    #        print(f"Shape after {layer.__class__.__name__}: {x.shape}")
    #    return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




###############################
####Data processing############
###############################
class ImageDataset(data.Dataset):
    def __init__(self, ann_file, transform=None):
        self.ann_file = ann_file
        self.transform = transform
        self.init()

    def init(self):
        
        self.im_names = []
        self.targets = []
        with open(self.ann_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(' ')
                self.im_names.append(data[0])
                self.targets.append(int(data[1]))

    def __getitem__(self, index):
        im_name = self.im_names[index]
        target = self.targets[index]

        img = Image.open(im_name).convert('RGB') 
        if img is None:
            print(im_name)
        
        img = self.transform(img)

        return img, img

    def __len__(self):
        return len(self.im_names)


def train_loader(train_list, batch_size):

    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        #transforms.RandomGrayscale(p=0.2),
        #transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    train_trans = transforms.Compose(augmentation)

    train_dataset = ImageDataset(train_list, transform=train_trans)   
    
    #we don't use parallel at this moment: 
    #We don't know how to do so for the evolution optimization
    train_sampler = None    

    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    shuffle=(train_sampler is None),
                    batch_size=batch_size,
                    num_workers=1,
                    pin_memory=True,
                    sampler=train_sampler,
                    drop_last=(train_sampler is None))

    return train_loader


def val_loader(val_list, batch_size):

    val_trans = transforms.Compose([
                    transforms.Resize(256),                   
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                  ])

    val_dataset = ImageDataset(val_list, transform=val_trans)      

    val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    shuffle=False,
                    batch_size=batch_size,
                    num_workers=1,
                    pin_memory=True)

    return val_loader


def adjust_learning_rate_cosine(optimizer, original_lr, epoch, num_epochs):
    """cosine learning rate annealing without restart"""
    lr = original_lr * 0.5 * (1. + math.cos(math.pi * epoch / num_epochs))

    current_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return current_lr


def save_checkpoint(state, filename):
    torch.save(state, filename)


def evaluate_model(val_list, model, batch_size, epoch, iteration, save_dir='./saved_img/'):
  
    # Remove previous images if the folder exists
    if epoch == 0:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    
    # Load validation data
    data_loader = val_loader(val_list, batch_size)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        images_saved = 0
        large_images_saved = 0
        
        for i, (input, target) in enumerate(data_loader):
            
            # Move input to GPU
            input = input.cuda(non_blocking=True)

            # Forward pass: generate the output (reconstruction) from the model
            output = model(input)

            # Move the input and output back to CPU for saving
            input_images = input.cpu()
            output_images = output.cpu()

            # Convert tensors to PIL images
            input_imgs = [transforms.ToPILImage()(img.squeeze()) for img in input_images]
            output_imgs = [transforms.ToPILImage()(img.squeeze()) for img in output_images]

            # Save 8 image pairs (real and reconstructed) in a large image
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            for idx in range(min(8, len(input_imgs))):
                # Real images on the left columns
                row = idx // 2
                col = (idx % 2) * 2
                axes[row, col].imshow(input_imgs[idx])
                axes[row, col].set_title(f"Real Image {images_saved + idx + 1}")
                axes[row, col].axis('off')
            
                # Reconstructed images on the right columns
                axes[row, col+1].imshow(output_imgs[idx])
                axes[row, col+1].set_title(f"Generated Image {images_saved + idx + 1}")
                axes[row, col+1].axis('off')

            # Save the grid image to the directory
            fig.suptitle(f'Epoch {epoch} - Image Pairs (Set {large_images_saved + 1})')
            fig.tight_layout()
            plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_iteration_{iteration}_image_pairs_set_{large_images_saved + 1}.png'))
            plt.close(fig)

            images_saved += len(input_imgs)
            large_images_saved += 1

            # If we've saved 10 large images (80 test images), break the loop
            if large_images_saved >= 10:
                break

# crop center region for masked back-propagation
def get_center_crop(tensor, size=64):
    _, _, H, W = tensor.shape
    start_h = (H - size) // 2
    start_w = (W - size) // 2
    return tensor[:, :, start_h:start_h+size, start_w:start_w+size]



###############################
#### Training #################
###############################
if __name__ == "__main__":

    original_lr = 0.05
    original_momentum = 0.9
    original_weight_decay = 1e-4
    original_batch_size = 32
    num_epochs = 10
    latent_dim = 128
    save_dir_img = './saved_images_main_7_Oct_10_2024/'
    save_path = "../../model_bin/main_7_Oct_10_2024/" # Define the path where you want to save the checkpoints
    log_dir = "./log_main_7_Oct_10_2024/" # Ensure the log directory exists
    train_file_list = './list/imagenet_full_train_list.txt'
    val_file_list = './list/imagenet_full_val_list.txt'


    # Remove the directory if it already exists
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)


    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    # Set up logging configuration
    log_file_path = os.path.join(log_dir, 'training_log.txt')
    logging.basicConfig(filename=log_file_path, 
                        level=logging.INFO, 
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    model = Autoencoder(latent_dim=latent_dim)

    # Define the device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to the selected device (GPU or CPU)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))


    optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=original_lr,
            momentum=original_momentum,
            weight_decay=original_weight_decay)
    
    train_loader = train_loader(train_file_list, original_batch_size)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(num_epochs):

        model.train()

        current_lr = adjust_learning_rate_cosine(optimizer, original_lr, epoch, num_epochs)
        print('current learning rate', epoch, current_lr)
        
        for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} Training"):
            
            # Move input and target to the device (GPU if available, otherwise CPU)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # In the training loop:
            output = model(input)

            # Create a mask for the central region
            mask = torch.zeros_like(output)
            _, _, H, W = output.shape
            start_h = (H - 64) // 2
            start_w = (W - 64) // 2
            mask[:, :, start_h:start_h+64, start_w:start_w+64] = 1

            # Apply the mask to both output and target
            output_masked = output * mask
            target_masked = target * mask

            # Calculate loss on the masked tensors
            loss = criterion(output_masked, target_masked)

            # compute gradient and do solver step
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update weights
            optimizer.step()

            # Log loss every 10 steps
            if i % 10 == 0:
                log_entry = f"Epoch {epoch}, Step {i}, Loss: {loss.item()}, Learning Rate: {current_lr}"
                logging.info(log_entry)  # Log to file

            if (i+1) % 5000 == 0:
                evaluate_model(val_list=val_file_list, model=model, batch_size=8, epoch=epoch, 
                               iteration=(i+1), save_dir=save_dir_img)
                model.train()

        if (epoch+1) % 1 == 0:
            # Save checkpoint after each epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'current_lr': current_lr
            }
            
            # Save the checkpoint in the specified path
            checkpoint_filename = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(checkpoint, filename=checkpoint_filename)

        evaluate_model(val_list=val_file_list, model=model, batch_size=8, epoch=epoch, 
                       iteration=len(train_loader), save_dir=save_dir_img)



