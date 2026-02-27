import os
import ifcb
import numpy as np
from PIL import Image
import argparse

DATA_DIR = '/Users/pao/Documents/CS199/real_data/Iligan - Copy/01-09-2023/Run 2/2025/D20250617'
OUTPUT_DIR = "./output_pngs"

def extract_images(data_dir,output_dir):
    try:
        data_dir = ifcb.DataDirectory(data_dir)
    except FileNotFoundError:
        print(f"Error: Data directory not found at '{data_dir}'.")
    except Exception as e:
        print(f"Error loading data directory: {e}")
        
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}")


    for sample_bin in data_dir:
        try:
            os.makedirs(f"{output_dir}/{sample_bin.pid}", exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}")
        for number, image in sample_bin.images.items():
            Image.fromarray(image.astype(np.uint8)*255).save(f"{output_dir}/{sample_bin.pid}/{number}.png")
    
    print("Saved all images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract IFCB images to PNGs")
    parser.add_argument("data_directory",help="Path to the directory containing IFCB data.")
    parser.add_argument("output_directory", help="Path to the directory to save the output CSV file and blobs.")

    args = parser.parse_args()

    extract_images("/Users/pao/Documents/CS199/real_data/Iligan - Copy/01-09-2023/Run 2/2025/D20250617","output_pngs")