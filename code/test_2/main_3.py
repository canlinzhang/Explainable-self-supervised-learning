import os
import torch
import numpy as np
import shutil
import math
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
import argparse
import gc

from functools import reduce
from torchvision import datasets, transforms
from collections import defaultdict, Counter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

#####################################
#### Network structure ##############
# Define the encoder
# Define the encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 32, 14, 14]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 32, 7, 7]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Tanh()  # Output range is now between -1 and 1
        )

    def quantize(self, z):
        # Quantization to 32 levels between 0 and 1
        z = torch.clamp(z, 0, 1)  # Ensure the values are between 0 and 1
        z_quantized = torch.round(z * 31) / 31  # Scale to 31, round, then scale back to [0, 1]
        return z_quantized

    def forward(self, x):
        z = self.encoder(x)
        z_quantized = self.quantize(z)
        return z_quantized


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



###############################
#### Parameter updating #######

#class function to obtain index for parameters
class ObtainIndexForParameters():
    '''
    The idea is:
    1. Obtain the layer name and shape of each layer.
    2. For each layer, obtain the original index range.
    3. Build index list for shuffle.
    4. In optimization, given an index, use index range to locate its layer and position.
    '''
    def __init__(self, autoencoder):

        self.autoencoder = autoencoder

    #obtain the layer name for encoder and decoder
    #also get the ID range for each layer
    #original index range:
    #Layer 1: ID1 to ID2
    #Layer 2: ID2+1 to ID3, etc.
    def obtain_layer_name(self):
        Dict = defaultdict()
        ID_s = 0
        for name, param in self.autoencoder.named_parameters():
            # Modify the name to remove the double 'encoder' or 'decoder' prefix
            name_parts = name.split('.')
            if name_parts[0] == 'encoder':
                # Remove the first 'encoder' and retain the rest
                name = '.'.join(name_parts[1:])
            elif name_parts[0] == 'decoder':
                # Remove the first 'decoder' and retain the rest
                name = '.'.join(name_parts[1:])
            
            List = list(param.shape)
            range = math.prod(List)
            Dict[name] = dict()
            Dict[name]['layer_shape'] = List
            Dict[name]['ID_range'] = [ID_s, ID_s + range - 1]
            ID_s += range

        #return both number of IDs and dictionary
        return(Dict, ID_s)


#class function to update autoencoder parameters
class UpdateParameters():

    def __init__(self, layer_dict):

        self.layer_dict = layer_dict

    #given an index, we will locate its corresponding layer.
    def locate_layer_by_id(self, ID):
        for name in self.layer_dict:
            if self.layer_dict[name]['ID_range'][0] <= ID <= self.layer_dict[name]['ID_range'][1]:
                return name
        raise ValueError('ID out of range: do not match the parameter number of the AE')

    def calculate_position_from_id(self, layer_name, tensor_shape, relative_id):
        """
        Calculate the position in a tensor given a query ID.
        
        Args:
            layer_name (str): Name of the layer (not used in calculation but kept for consistency)
            tensor_shape (list): Shape of the tensor [N, C, H, W] or [D] for one-dimensional tensors
            relative_id (int): relative parameter position
        
        Returns:
            list: Position corresponding to the query_id
        """
        # Use relative ID to locate the parameter position
        if len(tensor_shape) == 4:
            # Get dimensions for CNN layer
            N, C, H, W = tensor_shape
            # Calculate position
            # The order is from inner to outer dimension: W -> H -> C -> N
            w = relative_id % W
            h = (relative_id // W) % H
            c = (relative_id // (W * H)) % C
            n = (relative_id // (W * H * C)) % N
            return [n, c, h, w]
        elif len(tensor_shape) == 1:
            # Get dimension for one-dimensional tensor (e.g., bias or batch norm)
            D = tensor_shape[0]
            # Calculate position
            d = relative_id % D
            return [d]
        else:
            raise ValueError("Unsupported tensor shape: {}".format(tensor_shape))

    def locate_and_calculate_position(self, query_id):
        """
        Locate the layer and calculate the position in the tensor for a given ID.
        
        Args:
            layer_dict (dict): Dictionary containing layer information.
            query_id (int): The ID to locate and convert to position.
        
        Returns:
            tuple: (layer_name, position) corresponding to the query_id
        """
        # Locate the layer name using the query ID
        layer_name = self.locate_layer_by_id(query_id)
        
        # Get the ID range for the located layer
        start_id = self.layer_dict[layer_name]['ID_range'][0]
        # Calculate the actual ID within the layer's range
        relative_id = query_id - start_id
        
        # Calculate the position in the tensor
        tensor_shape = self.layer_dict[layer_name]['layer_shape']
        position = self.calculate_position_from_id(layer_name, tensor_shape, relative_id)
        
        return layer_name, position

    def get_parameter_value(self, autoencoder, layer_name, position):
        """
        Get the value of the parameter given the layer name and position.

        Args:
            layer_name (str): The name of the layer.
            position (list): The position of the parameter in the tensor.

        Returns:
            float: The value of the parameter at the specified position.
        """
        # Find the parameter tensor in the autoencoder
        for name, param in autoencoder.named_parameters():
            if name.endswith(layer_name):
                # Convert position list to a tuple for indexing
                pos_tuple = tuple(position)
                return param.data[pos_tuple].item()
        raise ValueError("Layer not found in the autoencoder: {}".format(layer_name))

    def set_parameter_value(self, autoencoder, layer_name, position, new_value):
        """
        Set the value of the parameter given the layer name and position.

        Args:
            layer_name (str): The name of the layer.
            position (list): The position of the parameter in the tensor.
            new_value (float): The new value to set.

        """
        # Find the parameter tensor in the autoencoder
        for name, param in autoencoder.named_parameters():
            if name.endswith(layer_name):
                # Convert position list to a tuple for indexing
                pos_tuple = tuple(position)
                param.data[pos_tuple] = new_value
                return
        raise ValueError("Layer not found in the autoencoder: {}".format(layer_name))

    def update_parameters_by_indices(self, autoencoder, indices, gamma):
        """
        Update the parameters corresponding to a list of indices.
        
        Args:
            indices (list): List of parameter indices to update.
            gamma (float): Threshold for random uniform number.
        
        Returns:
            autoencoder: The updated autoencoder model.
        """
        for query_k in indices:
            # Locate layer and position
            layer_name, position = self.locate_and_calculate_position(query_k)
            
            # Get current value
            value_k = self.get_parameter_value(autoencoder, layer_name, position)
            
            # Obtain random uniform number between (-gamma, gamma)
            kappa_k = random.uniform(-gamma, gamma)
            
            # Update parameter value
            self.set_parameter_value(autoencoder, layer_name, position, value_k + kappa_k)
        
        return autoencoder


# Define a class to calculate encoder non-zero dimensions and information captured
class AutoencoderEvaluator:
    def __init__(self, alpha, M):
        # the final return is:
        # captured information * alpha + compression_rate
        # We may keep alpha = 0 for just focus on compression rate
        self.alpha = alpha
        self.M = M

    #we may or may not use this func !!!!!
    def count_non_zero_dimensions(self, encoder_output):
        # Count non-zero values in each (num_filter, h, w) tensor along the channel axis
        batch_size = encoder_output.shape[0]
        non_zero_count = []
        zero_count = []
        for i in range(batch_size):
            non_zero_dims = torch.sum(encoder_output[i] != 0, dim=(1, 2))  # Count non-zero values for each channel (dim 1)
            total_non_zero = torch.sum(non_zero_dims)  # Sum across all channels to get total non-zero values
            total_zero = encoder_output[i].numel() - total_non_zero  # Total elements minus non-zero elements
            non_zero_count.append(total_non_zero.item())
            zero_count.append(total_zero.item())
        return non_zero_count, zero_count

    def calculate_information(self, encoder_output, num_bins=32):
        """
        Calculate the amount of information required to represent each batch's encoder output.

        Args:
            encoder_output (torch.Tensor): The output of the encoder with shape (batch_size, num_filters, h', w').
            num_bins (int): The number of quantized levels (default is 32).

        Returns:
            information_list (list): A list where each element is the amount of information for a batch sample.
        """
        # Flatten the entire encoder output to calculate the global probability distribution
        flattened_output = encoder_output.flatten()  # Shape: (batch_size * num_filters * h' * w',)

        # Scale the values to range [0, num_bins - 1] and convert to integer indices
        quantized_indices = (flattened_output * (num_bins - 1)).round().to(torch.int64)

        # Calculate histogram to get counts of each value (0, 1, ..., num_bins-1)
        histogram = torch.bincount(quantized_indices, minlength=num_bins).float()
        total_elements = flattened_output.numel()

        # Calculate probabilities (p0, p1, ..., p31)
        probabilities = histogram / total_elements  # Shape: (num_bins,)

        # Avoid log(0) by masking zero probabilities
        probabilities[probabilities == 0] = 1e-12  # Add a small value to avoid log(0)

        # Compute -log2(pi) for all probabilities
        log_probabilities = -torch.log2(probabilities)

        # Reshape the encoder output back to batch-wise dimensions
        batch_size, num_filters, h_prime, w_prime = encoder_output.shape

        # Calculate information for each batch
        information_list = []
        for i in range(batch_size):
            # Scale the batch values to quantized indices
            batch_quantized_indices = (encoder_output[i].flatten() * (num_bins - 1)).round().to(torch.int64)

            # Count the occurrences of each quantized value in the current batch
            batch_histogram = torch.bincount(
                batch_quantized_indices, minlength=num_bins
            ).float()

            # Compute the total information for the current batch
            batch_information = torch.sum(batch_histogram * log_probabilities).item()
            information_list.append(batch_information)

        return information_list

    def evaluate_batch(self, encoder_output, original_images, reconstructed_images):
        """
        Compute the information compression rate of the autoencoder.

        Args:
            encoder_output (torch.Tensor): Encoder output with shape (batch_size, num_filters, h', w').
            original_images (torch.Tensor): Original images with shape (batch_size, 1, H, W).
            reconstructed_images (torch.Tensor): Reconstructed images with shape (batch_size, 1, H, W).

        Returns:
            float: The final compression rate score.
        """
        # Calculate information in original images
        information_original = self.calculate_information(original_images, num_bins=256)

        # Calculate information in encoder output
        information_encoder = self.calculate_information(encoder_output, num_bins=32)

        # Compute residual images
        residual_images = original_images - reconstructed_images

        # Normalize residual images to range [0, 1]
        residual_images = (residual_images + 1) / 2

        # Calculate information in residual images
        information_residual = self.calculate_information(residual_images, num_bins=512)

        # Calculate information obtained by reconstruction
        information_reconstruct = [max(0, orig - resid) for orig, resid in zip(information_original, information_residual)]

        # Calculate compression rate for each batch
        compress_batch = [reconstruct / (encoder + 1e-12) for reconstruct, encoder in zip(information_reconstruct, information_encoder)]

        # Sort and take the top M compression rates
        top_compression_rates = sorted(compress_batch, reverse=True)[:self.M]

        # Return the final compression rate score
        final_score = sum(top_compression_rates)
        return final_score, information_original, information_residual, information_reconstruct, information_encoder


'''
First, get layer_dict and number of total parameters by
helper = ObtainIndexForParameters(autoencoder)
layer_dict, total_ids = helper.obtain_layer_name()

Then, build an index List = [0, 1, 2, ..., total_ids-1] and shuffle it. 
Then, segment List into small segments, with each segments contains K indices (the last segment contains whatever remains).

In each step, we update the autoencoder according to the segment. 
That is, we first read the autoencoder parameter from the optimal autoencoder. 
Then, given the current segment, we update the autoencoder according to each of its index:

autoencoder = updater.update_parameters_by_indices(autoencoder, segments, gamma)

With the updated autoencoder, implement it on N randomly selected MNIST training (or validation) images. 

If the target score increases (for now you only need to know there is such a score. 
You may use score = F(autoencoder, sampled images) to hold a space in the pipeline ), 
we over-write the optimal autoencoder parameters by the current updated autoencoder. 

Then, move to the next step. Do this until the for loop ends.
'''
class AutoencoderOptimizer:
    def __init__(self, autoencoder, K=20, N=2000, M=100, gamma=0.1, alpha=0.0, 
                 dataset_path='../../data/MNIST/',
                 save_dir='../../model_bin/test_2_main_1_Dec_7_2024/'):
        self.autoencoder = autoencoder
        self.K = K
        self.N = N
        self.gamma = gamma
        self.alpha = alpha
        self.dataset_path = dataset_path
        self.save_dir = save_dir
        self.M = M
        
        # Load latest checkpoint if available
        self.start_epoch = 0
        if os.path.exists(self.save_dir):
            checkpoint_files = [f for f in os.listdir(self.save_dir) if f.startswith('optimal_autoencoder_model_epoch_') and f.endswith('.pth')]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
                latest_checkpoint = checkpoint_files[0]
                checkpoint_path = os.path.join(self.save_dir, latest_checkpoint)
                self.autoencoder.load_state_dict(torch.load(checkpoint_path).state_dict())
                self.start_epoch = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
                print(f'Loaded checkpoint: {latest_checkpoint}')
        
        # Obtain layer_dict and number of total parameters
        self.helper = ObtainIndexForParameters(self.autoencoder)
        self.layer_dict, self.total_ids = self.helper.obtain_layer_name()
        
        # Initialize updater
        self.updater = UpdateParameters(self.layer_dict)

        # Create an evaluator instance
        self.evaluator = AutoencoderEvaluator(self.alpha, self.M)

        # Build and shuffle index list
        self.index_list = list(range(self.total_ids))
        random.shuffle(self.index_list)
        
        # Segment the index list into smaller segments of size K
        self.segments = [self.index_list[i:i + self.K] for i in range(0, len(self.index_list), self.K)]

        print('original indices', self.index_list[:10], self.segments[0])
        
        # Load MNIST dataset for sampling training/validation images
        transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.MNIST(root=self.dataset_path, train=True, download=True, transform=transform)
        
        # Split dataset into training and validation
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
        
        # Create DataLoader to randomly sample N images
        self.data_loader = DataLoader(self.train_dataset, batch_size=self.N, shuffle=True)
        
        # Assume that the optimal autoencoder is initially the same as the original
        self.optimal_autoencoder = copy.deepcopy(self.autoencoder)
        self.optimal_score = float('-inf')  # Initialize optimal score as negative infinity to maximize the negative MSE

        self.decay_rate = 0.0

    def optimize(self, num_epochs):
        # Move optimal autoencoder to GPU before starting optimization
        self.optimal_autoencoder = self.optimal_autoencoder.to('cuda')

        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            
            # Iterate through each segment and update autoencoder
            with tqdm(range(len(self.segments)), desc='optimizing', unit='segment') as pbar:
                for i_1 in pbar:
                    torch.cuda.empty_cache()  # Free GPU memory to avoid OOM issues if necessary

                    segment = self.segments[i_1]

                    # deepcopy to make the optimal AE for use in this step. Otherwise update parameter will change
                    # the parameters of the optimal AE !!!
                    optimal_auto_temp = copy.deepcopy(self.optimal_autoencoder)

                    # Update autoencoder parameters according to the current segment
                    autoencoder = self.updater.update_parameters_by_indices(optimal_auto_temp, segment, self.gamma)

                    # Move updated autoencoder to GPU for forward pass
                    autoencoder = autoencoder.to('cuda')

                    # Randomly sample N images from the MNIST training dataset
                    sampled_images, sampled_labels = next(iter(self.data_loader))

                    # Move sampled images to GPU for forward pass
                    sampled_images_gpu = sampled_images.to('cuda')

                    # Calculate the accuracy for the updated autoencoder with the sampled images
                    # Use the autoencoder to reconstruct the sampled images
                    with torch.no_grad():
                        encoder_current = autoencoder.encoder(sampled_images_gpu)  # Runs on GPU
                        reconstructed_current = autoencoder(sampled_images_gpu)    # Runs on GPU

                        # Move tensors back to CPU for evaluation
                        encoder_current = encoder_current.cpu()
                        reconstructed_current = reconstructed_current.cpu()

                    # Evaluate on CPU using the original sampled_images
                    final_current, l1, l2, l3, l4 = self.evaluator.evaluate_batch(encoder_current, sampled_images, reconstructed_current)

                    # Calculate the accuracy for the optimal autoencoder with the sampled images
                    with torch.no_grad():
                        encoder_optimal = self.optimal_autoencoder.encoder(sampled_images_gpu)  # Runs on GPU
                        reconstructed_optimal = self.optimal_autoencoder(sampled_images_gpu)    # Runs on GPU

                        # Move tensors back to CPU for evaluation
                        encoder_optimal = encoder_optimal.cpu()
                        reconstructed_optimal = reconstructed_optimal.cpu()

                    # Evaluate on CPU using the original sampled_images
                    final_optimal, l1_, l2_, l3_, l4_ = self.evaluator.evaluate_batch(encoder_optimal, sampled_images, reconstructed_optimal)

                    # Update progress bar with accuracy
                    pbar.set_postfix({'cur_s': final_current, 
                                      'opt_s': final_optimal})

                    # If the current autoencoder performs better, update the optimal autoencoder
                    if final_current > final_optimal:
                        self.optimal_autoencoder.load_state_dict(autoencoder.state_dict())
                        self.optimal_score = final_current

                    del(optimal_auto_temp, autoencoder)
                    torch.cuda.empty_cache()  # Free GPU memory if needed

            self.index_list = list(range(self.total_ids))
            random.shuffle(self.index_list)
            
            # Segment the index list into smaller segments of size K
            self.segments = [self.index_list[i:i + self.K] for i in range(0, len(self.index_list), self.K)]

            print('indices', self.index_list[:10], self.segments[0])

            # Ensure the directory exists before saving the model
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            torch.save(self.optimal_autoencoder, f'{self.save_dir}optimal_autoencoder_model_epoch_{epoch}.pth')

        # Move the optimal autoencoder back to CPU after all epochs are complete
        self.optimal_autoencoder = self.optimal_autoencoder.to('cpu')

        print("Optimization complete. Best accuracy: {:.3f}%".format(self.optimal_score))



##########################
'''
Do

python main_3.py --mode train
python main_3.py --mode eval

to train or evaluate the model
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or evaluate the autoencoder model.')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, help="Mode of operation: 'train' or 'eval'")
    args = parser.parse_args()

    if args.mode == 'train':
        autoencoder = Autoencoder()

        # Example usage
        autoencoder_optimizer = AutoencoderOptimizer(autoencoder)
        autoencoder_optimizer.optimize(20)

    elif args.mode == 'eval':
        # Load a saved autoencoder model to evaluate
        checkpoint_path = '../../model_bin/test_2_main_1_Dec_7_2024/optimal_autoencoder_model_epoch_49.pth'  # Replace '9' with the epoch number you want to load
        autoencoder = torch.load(checkpoint_path)
        autoencoder = autoencoder.to('cpu')  # Move to CPU
        autoencoder.eval()  # Set to evaluation mode

        # Load MNIST dataset for evaluation
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST(root='../../data/MNIST/', train=False, download=True, transform=transform)
        data_loader = DataLoader(dataset, batch_size=10, shuffle=True)  # Load a small batch for visualization

        # Evaluate and save reconstructed images
        save_image_dir = './saved_images_3/'
        os.makedirs(save_image_dir, exist_ok=True)

        sampled_images, _ = next(iter(data_loader))
        with torch.no_grad():
            reconstructed_images = autoencoder(sampled_images)

        # Save the original and reconstructed images
        for i in range(sampled_images.size(0)):
            fig, axes = plt.subplots(1, 2)
            # Original image
            axes[0].imshow(sampled_images[i].squeeze().numpy(), cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            # Reconstructed image
            axes[1].imshow(reconstructed_images[i].squeeze().numpy(), cmap='gray')
            axes[1].set_title('Reconstructed Image')
            axes[1].axis('off')
            
            # Save the figure
            plt.savefig(f'{save_image_dir}reconstructed_image_{i}.png')
            plt.close(fig)





