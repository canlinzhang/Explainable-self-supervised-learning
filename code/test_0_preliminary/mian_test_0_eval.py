import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image


from main_test_0 import Autoencoder  # Assuming the Autoencoder class is defined here

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

def run_inference_and_visualize(model, dataloader, device, output_dir='./output_images/', display_images=True):
    
    model.eval()

    # Function to reverse normalization and clamp values
    def unnormalize_and_clamp(image):
        # Reverse normalization
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        # Clamp values to ensure they are within [0, 1]
        return image.clamp(0, 1)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(dataloader, desc="Processing")):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Save output images
            for i, output in enumerate(outputs):
                image_filename = f'output_{batch_idx * dataloader.batch_size + i}.png'
                save_image(output, os.path.join(output_dir, image_filename))

                # Visualization
                if display_images:
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(unnormalize_and_clamp(inputs[i].cpu()).permute(1, 2, 0))
                    plt.title('Input Image')
                    plt.subplot(1, 2, 2)
                    plt.imshow(unnormalize_and_clamp(outputs[i].cpu()).permute(1, 2, 0))
                    plt.title('Output Image')
                    plt.show()

def prepare_data_and_run_inference(model_path, folder_path, device, transform):
    # Load the model
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare the data
    dataset = UnlabeledDataset(folder_path=folder_path, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)  # Set batch size to 1 for easier visualization
    
    # Run inference
    run_inference_and_visualize(model, loader, device)


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformations (include normalization as used during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Call the function
model_path = '../model_bin/main_test_1_Aug_30_2024/autoencoder_epoch_6.pth'  # Set the path to your model file
folder_path = '../data/train_combo/'  # Set the path to your validation data
prepare_data_and_run_inference(model_path, folder_path, device, transform)

