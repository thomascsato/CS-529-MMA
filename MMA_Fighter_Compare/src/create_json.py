"""
This script aims to get a list of names from an images folder that I have so
that I can easily access it when searching in the front end of my application.
As well as the fighter names.
"""

import os
import json

def create_json_from_images(directory, output_path):
    # List to store the formatted data
    data = []
    # List to store just the fighter names
    names = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a PNG image
        if filename.endswith('.png'):
            # Split the filename to get the name without extension
            name = filename.replace('_', ' ').replace('.png', '')
            # Append the dictionary to the data list
            data.append({"name": name, "url": filename})
            # Append the name to the names list
            names.append(name)

    # Define the output JSON file paths
    output_data_file = os.path.join(output_path, 'fighters.json')
    output_names_file = os.path.join(output_path, 'fighter_names.json')

    # Write the list to a JSON file for full data
    with open(output_data_file, 'w') as json_file:
        json.dump(data, json_file, indent=2)

    # Write the list of names to a separate JSON file
    with open(output_names_file, 'w') as json_file:
        json.dump(names, json_file, indent=2)

    print(f'JSON files created at {output_data_file} and {output_names_file}')

# Specify the directory containing images
image_directory = 'C:\\Users\\thoma\\OneDrive\\Documents\\CS 529\\mma-angular-app\\public'
output_directory = 'C:\\Users\\thoma\\OneDrive\\Documents\\CS 529\\mma-angular-app\\src\\app\\components\\image-selector'

# Call the function to create JSON from images
create_json_from_images(image_directory, output_directory)