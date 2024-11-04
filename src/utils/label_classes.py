import os
import numpy as np

# Step 2: Define the directory path
directory_path = 'E:\Projects\Sign Language Project\ASL/train_reduced100'

# Step 3: List the directories
all_entries = os.listdir(directory_path)

# Step 4: Filter out non-directory files
label_classes = [entry for entry in all_entries if os.path.isdir(
    os.path.join(directory_path, entry))]

# Step 5: Save the list of directories to a .npy file
np.save('label_classes.npy', label_classes)
