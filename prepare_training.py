"""
Prepare Training Data Script

This script:
1. Splits YOLO dataset into train/val
2. Creates data.yaml automatically

Dataset structure expected:

dataset/
│
├── images/
└── labels/

Negative images are supported.
"""

import os
import random
import shutil
from tqdm import tqdm
import yaml


def split_data(input_folder, output_folder, train_ratio=0.8):

    images_folder = os.path.join(input_folder, 'images')
    labels_folder = os.path.join(input_folder, 'labels')

    if not os.path.exists(images_folder):
        raise ValueError(f"Images folder not found: {images_folder}")

    os.makedirs(output_folder, exist_ok=True)

    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')

    for folder in [train_folder, val_folder]:
        os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(folder, 'labels'), exist_ok=True)

    image_files = [
        f for f in os.listdir(images_folder)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ]

    print(f"\nFound {len(image_files)} images")

    random.shuffle(image_files)

    train_count = int(len(image_files) * train_ratio)

    train_images = image_files[:train_count]
    val_images = image_files[train_count:]

    def copy_files(file_list, split_name):

        for img_file in tqdm(file_list, desc=f"Copying {split_name}"):

            label_file = os.path.splitext(img_file)[0] + ".txt"

            src_img = os.path.join(images_folder, img_file)
            dst_img = os.path.join(output_folder, split_name, 'images', img_file)

            shutil.copy(src_img, dst_img)

            # Copy label only if it exists
            src_lbl = os.path.join(labels_folder, label_file)

            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(output_folder, split_name, 'labels', label_file)
                shutil.copy(src_lbl, dst_lbl)

    copy_files(train_images, "train")
    copy_files(val_images, "val")

    print("\nDataset split completed!")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")


def create_data_yaml(classes_txt_path, yaml_output_path):

    with open(classes_txt_path, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    data = {
        'path': './split',
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(classes),
        'names': classes
    }

    with open(yaml_output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"\ndata.yaml created at: {yaml_output_path}")


if __name__ == "__main__":

    # YOUR EXPORTED YOLO DATASET FOLDER
    datapath = "./dataset"

    # OUTPUT SPLIT FOLDER
    outputpath = "./split"

    # SPLIT DATASET
    split_data(datapath, outputpath, train_ratio=0.9)

    # CREATE YAML
    create_data_yaml(
        "./dataset/classes.txt",
        "./data.yaml"
    )

    print("\nEverything completed successfully!")

