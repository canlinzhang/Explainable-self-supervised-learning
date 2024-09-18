
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


# Define your model architecture here
class YourModel(torch.nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Define layers here

    def forward(self, x):
        # Define forward pass here
        return x
  

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


def evaluate_model(val_list, model, batch_size, file_name, save_dir='./saved_images_main_3_Sept_17_2024/'):
  
    # Load validation data
    data_loader = val_loader(val_list, batch_size)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        images_saved = 0
        for i, (input, target) in enumerate(data_loader):
            
            # Move input to GPU
            input = input.cuda(non_blocking=True)

            # Forward pass: generate the output (reconstruction) from the model
            output = model(input)

            # Move the input and output back to CPU for saving
            input_images = input.cpu()
            output_images = output.cpu()

            # Save 8 image pairs (real and reconstructed)
            if images_saved < 8:
                # Convert tensors to PIL images
                input_imgs = [transforms.ToPILImage()(img.squeeze()) for img in input_images[:min(8-images_saved, batch_size)]]
                output_imgs = [transforms.ToPILImage()(img.squeeze()) for img in output_images[:min(8-images_saved, batch_size)]]

                # Save the pairs in a grid (2 rows, 4 columns)
                fig, axes = plt.subplots(2, 4, figsize=(12, 6))
                for idx in range(len(input_imgs)):
                    # Real images on the left columns
                    row = idx // 4
                    col = idx % 4
                    axes[row, col].imshow(input_imgs[idx])
                    axes[row, col].set_title(f"Real Image {idx+1}")
                    axes[row, col].axis('off')
                
                for idx in range(len(output_imgs)):
                    # Reconstructed images on the right columns
                    row = (idx // 4)
                    col = idx % 4
                    axes[row, col].imshow(output_imgs[idx])
                    axes[row, col].set_title(f"Generated Image {idx+1}")
                    axes[row, col].axis('off')

                # Save the grid image to the directory
                fig.suptitle(f'image comparison')
                fig.tight_layout()
                plt.savefig(os.path.join(save_dir, file_name))
                plt.close(fig)

                images_saved += len(input_imgs)

            if images_saved >= 8:
                break


# Load the models from their respective .pth files
checkpoint_2 = torch.load('../../model_bin/Sept_17_2024/checkpoint_epoch_100.pth')
model_2 = YourModel()
model_2.load_state_dict(checkpoint_2['model_state_dict'])

# Ensure the models are in evaluation mode
model_2.eval()


# Evaluate both models
#evaluate_model('./list/caltech101_list.txt', model_2, 8, 'my_code.png')






