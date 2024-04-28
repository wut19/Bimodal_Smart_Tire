"""
    Organize data w.r.t. visual modality
"""

import os
import shutil

def restructure_files(source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Traverse through each subdirectory in the source directory
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            # Get the full path of the subdirectory
            sub_dir = os.path.join(root, dir_name)

            # get subdirectories
            sub_dirs_inside = [name for name in os.listdir(sub_dir) if os.path.isdir(os.path.join(sub_dir, name))]
            for sub_sub_dir in sub_dirs_inside:
                os.makedirs(os.path.join(destination_dir, sub_sub_dir, dir_name), exist_ok=True)
                src = os.path.join(sub_dir, sub_sub_dir)
                dest = os.path.join(destination_dir, sub_sub_dir, dir_name)
                files = os.listdir(src)
                for file in files:
                    shutil.copy(os.path.join(src, file), os.path.join(dest, file))

# Example usage:
source_directory = ""
destination_directory = ""

restructure_files(source_directory, destination_directory)
