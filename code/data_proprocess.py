
from PIL import Image, ImageOps
import os
import shutil
from tqdm import tqdm
import argparse

'''Pre-process to remove keep only image file in the data.
That is, we want
$your_dataset_path
    |──class1
        |──xxxx.jpg
        |──...
    |──class2
        |──xxxx.jpg
        |──...
    |──...
    |──classN
        |──xxxx.jpg
        |──...
We do this to use the training structure from https://github.com/Horizon2333/imagenet-autoencoder.
So, we will remove all other files under each class folder. Only keep jpg and jpeg file.
'''

def list_subfolders_with_path(root_folder):
    # List all entries in the directory given by "root_folder"
    entries = os.listdir(root_folder)
    # Filter entries to include only directories and join them with the root_folder path
    subfolders = [os.path.join(root_folder, entry) for entry in entries if os.path.isdir(os.path.join(root_folder, entry))]
    return subfolders


def clean_folder_keep_images(folder_directory):
  # Define the allowed image extensions
  allowed_extensions = ('.jpg', '.jpeg')

  # Iterate over all entries in the folder
  for entry in os.listdir(folder_directory):
      # Construct the full path of each entry
      entry_path = os.path.join(folder_directory, entry)
      
      if os.path.isdir(entry_path):
          # If entry is a directory, remove it and all its contents
          shutil.rmtree(entry_path)
          print('removed folder: ', entry_path)
      elif not entry.lower().endswith(allowed_extensions):
          # If the entry is a file and not an allowed image type, remove it
          os.remove(entry_path)
          print('removed file: ', entry)


def clean_up_training_data(root_folder):
    # First, remove all files (including hidden files) in the root folder
    for entry in os.listdir(root_folder):
        entry_path = os.path.join(root_folder, entry)
        if os.path.isfile(entry_path):
            os.remove(entry_path)
            print(f'Removed file from root: {entry_path}')

    # Now, process only the subfolders
    subfolders = [os.path.join(root_folder, d) for d in os.listdir(root_folder) 
                  if os.path.isdir(os.path.join(root_folder, d))]

    for subfolder in tqdm(subfolders, desc='clean sub-folders'):
        clean_folder_keep_images(subfolder)

    print(f'Cleaned up {len(subfolders)} subfolders in {root_folder}')


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Clean up training data folders.')
    parser.add_argument('root_folder', type=str, help='Path to the root folder containing training data subfolders')

    # Parse arguments
    args = parser.parse_args()

    # Run the clean-up function with the provided root folder
    clean_up_training_data(args.root_folder)


##########################
if __name__ == "__main__":
    main()



