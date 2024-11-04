import os
import shutil
import random


def reduce_dataset(original_dir, reduced_dir, samples_per_class):
    """
    Reduces the number of images per class by randomly sampling them.

    Parameters:
    - original_dir: Path to the original dataset directory.
    - reduced_dir: Path to the new directory where the reduced dataset will be saved.
    - samples_per_class: Number of images to keep per class.
    """
    if not os.path.exists(reduced_dir):
        os.makedirs(reduced_dir)

    # Loop over each class folder in the original directory
    for class_name in os.listdir(original_dir):
        class_dir = os.path.join(original_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Create a new folder for the class in the reduced dataset directory
        reduced_class_dir = os.path.join(reduced_dir, class_name)
        if not os.path.exists(reduced_class_dir):
            os.makedirs(reduced_class_dir)

        # Get all images in the class directory
        image_files = os.listdir(class_dir)

        # Randomly select a subset of images
        selected_images = random.sample(image_files, samples_per_class)

        # Copy the selected images to the reduced dataset directory
        for image_file in selected_images:
            src = os.path.join(class_dir, image_file)
            dst = os.path.join(reduced_class_dir, image_file)
            shutil.copy(src, dst)


# Paths to the original and reduced dataset directories
original_dataset_dir = 'E:/Projects/Sign Language Project/SignSpeak/data/raw/train'
reduced_dataset_dir = 'E:/Projects/Sign Language Project/SignSpeak/data/raw/train_reduced500'

# Number of images to keep per class
samples_per_class = 500  # Reduce to 1000 images per class

# Reduce the dataset
reduce_dataset(original_dataset_dir, reduced_dataset_dir, samples_per_class)
