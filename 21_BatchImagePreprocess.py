#!/usr/bin/env python
# coding: utf-8

# In[19]:


import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


input_dir = r'E:\Input_images'  # Update this path to your input directory
output_dir = r'E:\Output_images'  # Update this path to your desired output directory


# In[21]:


# Function to apply preprocessing steps
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Converted {image_path} to grayscale.")
    
    # Noise reduction using Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print(f"Applied Gaussian blur to {image_path}.")
    
    # Contrast enhancement using histogram equalization
    equalized = cv2.equalizeHist(blurred)
    print(f"Applied histogram equalization to {image_path}.")
    
    # Edge detection using Canny edge detector
    edges = cv2.Canny(equalized, 50, 150)
    print(f"Applied Canny edge detection to {image_path}.")
    
    return edges

# Function to process all images in a directory recursively
def process_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, file)
                print(f"Processing: {image_path}")
                
                # Apply preprocessing
                processed_image = preprocess_image(image_path)
                
                if processed_image is None:
                    print(f"Skipping image {image_path} due to loading error.")
                    continue
                
                # Construct the output file path
                relative_path = os.path.relpath(root, input_dir)
                save_dir = os.path.join(output_dir, relative_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, file)
                
                # Save the processed image
                cv2.imwrite(save_path, processed_image)
                print(f"Processed and saved: {save_path}")
                
                # Display the original and processed images for verification
                original_image = cv2.imread(image_path)
                plt.figure(figsize=(10, 5))
                
                plt.subplot(1, 2, 1)
                plt.title('Original Image')
                plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                
                plt.subplot(1, 2, 2)
                plt.title('Processed Image')
                plt.imshow(processed_image, cmap='gray')
                
                plt.show()

# Define input and output directories
#input_dir = '/mnt/data/input_images'  # Update this path to your input directory
#output_dir = '/mnt/data/output_images'  # Update this path to your desired output directory

# Process all images in the input directory recursively
process_directory(input_dir, output_dir)

