import os
from PIL import Image

# Function to check if a file is in JPG format
def is_jpg(file_path):
    try:
        img = Image.open(file_path)
        return img.format == 'JPEG'
    except Exception as e:
        return False

# Directory containing your image dataset folders
dataset_directory = "./Food2k_complete/"

# Output file to store paths of removed images
output_file = "removed_images.txt"

# List to store removed image file paths
removed_image_paths = []

# Loop through all folders in the dataset directory
for folder in os.listdir(dataset_directory):
    folder_path = os.path.join(dataset_directory, folder)
    
    # Check if it's a directory
    if os.path.isdir(folder_path):
        print(f"Checking folder: {folder}")
        
        # Loop through all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Check if it's a file and not in JPG format
            if os.path.isfile(file_path) and not is_jpg(file_path):
                print(f"Removing file not in JPG format: {file_path}")
                
                # Add the file path to the list of removed images
                removed_image_paths.append(file_path)
                
                # Remove the file
                os.remove(file_path)

# Write the removed image paths to the output file
with open(output_file, 'w') as f:
    for path in removed_image_paths:
        f.write(f"{path}\n")

print("Removed images and saved their paths to", output_file)
