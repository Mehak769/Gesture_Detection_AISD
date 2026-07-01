"""
Prepare Training Data Script

This script provides utilities for:
1. Splitting dataset into training and validation sets
2. Creating YOLO data configuration YAML file

Usage:
    # Set your paths and parameters
    datapath = "/path/to/your/dataset"
    outputpath = "./split"
    train_pct = 0.8
    
    # Split the data
    split_data(datapath, outputpath, train_ratio=train_pct)
    
    # Create YAML config
    create_data_yaml('path/to/classes.txt', 'path/to/data.yaml')
    
"""

import os
import random
import shutil
from tqdm import tqdm
import yaml


def split_data(input_folder, output_folder, train_ratio=0.8, val_ratio=0.2):
    """
    Splits data into training and validation sets.

    Args:
    - input_folder (str): Path to the folder containing the 'images' and 'labels' subfolders.
    - output_folder (str): Path where the train and val directories will be created.
    - train_ratio (float): Ratio of data used for training (default 0.8).
    - val_ratio (float): Ratio of data used for validation (default 0.2).
    """

    # Check if input folder exists
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder '{input_folder}' does not exist.")

    # Check if images and labels folders exist
    images_folder = os.path.join(input_folder, 'images')
    labels_folder = os.path.join(input_folder, 'labels')

    if not os.path.exists(images_folder):
        raise ValueError(f"Images folder '{images_folder}' does not exist.")
    if not os.path.exists(labels_folder):
        raise ValueError(f"Labels folder '{labels_folder}' does not exist.")

    # Check if output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create subfolders for train and val
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    # Create subfolders for images and labels inside train and val folders
    for subfolder in ['images', 'labels']:
        os.makedirs(os.path.join(train_folder, subfolder), exist_ok=True)
        os.makedirs(os.path.join(val_folder, subfolder), exist_ok=True)

    # Get all image files from the images folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]
    label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]

    # Debugging: Print out the files
    print(f"Found {len(image_files)} image files: {image_files}")
    print(f"Associated label files: {label_files}")

    # If no image files were found, raise an error
    if len(image_files) == 0:
        raise ValueError("No image files found in the specified directory.")

    # Shuffle the list of image files for random splitting
    combined = list(zip(image_files, label_files))
    if len(combined) == 0:
        raise ValueError("No valid image-label file pairs found.")

    random.shuffle(combined)
    image_files, label_files = zip(*combined)

    # Calculate number of training and validation samples
    total_count = len(image_files)
    train_count = int(total_count * train_ratio)
    val_count = total_count - train_count

    # Copy training data to train folder
    for i in tqdm(range(train_count), desc="Copying training data"):
        img_file = image_files[i]
        lbl_file = label_files[i]

        # Copy image
        shutil.copy(os.path.join(images_folder, img_file), os.path.join(train_folder, 'images', img_file))
        # Copy label
        shutil.copy(os.path.join(labels_folder, lbl_file), os.path.join(train_folder, 'labels', lbl_file))

    # Copy validation data to val folder
    for i in tqdm(range(train_count, total_count), desc="Copying validation data"):
        img_file = image_files[i]
        lbl_file = label_files[i]

        # Copy image
        shutil.copy(os.path.join(images_folder, img_file), os.path.join(val_folder, 'images', img_file))
        # Copy label
        shutil.copy(os.path.join(labels_folder, lbl_file), os.path.join(val_folder, 'labels', lbl_file))

    print(f"Data split complete: {train_count} images for training and {val_count} images for validation.")


def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    """
    Creates a YOLO data configuration YAML file from a classes.txt file.
    
    Args:
    - path_to_classes_txt (str): Path to the classes.txt file containing class names (one per line)
    - path_to_data_yaml (str): Path where the data.yaml file will be created
    """
    # Check if the classes.txt file exists
    if not os.path.exists(path_to_classes_txt):
        print(f'Error: {path_to_classes_txt} not found! Please ensure the classes.txt file exists.')
        return

    # Read class names from classes.txt
    with open(path_to_classes_txt, 'r') as f:
        classes = []
        for line in f.readlines():
            if len(line.strip()) == 0: 
                continue  # Ignore empty lines
            classes.append(line.strip())
    
    if not classes:
        print('Error: classes.txt is empty or no valid classes were found.')
        return

    # Get number of classes
    number_of_classes = len(classes)

    # Create the data dictionary
    data = {
        'path': './data',
        'train': 'train/images',
        'val': 'validation/images',
        'nc': number_of_classes,
        'names': classes
    }

    # Write the dictionary to a YAML file
    try:
        with open(path_to_data_yaml, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        print(f'Successfully created config file at {path_to_data_yaml}')
    except Exception as e:
        print(f'Error writing to {path_to_data_yaml}: {e}')



# ============================================================================
# Main execution section
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # Step 1: Split data into training and validation sets
    # ========================================================================
    
    # Set paths and parameters
    #datapath = "/path/to/your/dataset"  # Your dataset path
    datapath = '/home/aisd_user12/my_training_data'
    outputpath = "./split"  # Where you want to save the split data
    train_pct = 0.9  # Percentage of data used for training
    
    # Uncomment the line below to split your data
    split_data(datapath, outputpath, train_ratio=train_pct)
    
    # ========================================================================
    # Step 2: Create YOLO data configuration YAML file
    # ========================================================================
    
    # Define the paths
    #path_to_classes_txt = 'path/to/classes.txt'  # Adjust to your classes.txt path
    path_to_classes_txt = '/home/aisd_user12/my_training_data/classes.txt'
    #path_to_data_yaml = 'path/to/yolo_config.yaml'  # Adjust to where you want to save yolo_config.yaml
    path_to_data_yaml = '/home/aisd_user12/yolo_config.yaml'        
    # Uncomment the line below to create the YAML file
    create_data_yaml(path_to_classes_txt, path_to_data_yaml)
    
    # Display the contents of the generated YAML file
    # print('\nFile contents:\n')
    # try:
    #     with open(path_to_data_yaml, 'r') as f:
    #         print(f.read())
    # except FileNotFoundError:
    #     print('YAML file not found.')
    


