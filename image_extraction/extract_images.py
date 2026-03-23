import os
import ifcb
import numpy as np
from PIL import Image, ImageOps
import argparse

DATA_DIR = '/Users/pao/Documents/CS199/organized_data/'
OUTPUT_DIR = "./output_pngs"

def extract_images(data_dir,output_dir):
    try:
            os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}")

    total_images_saved = 0

    for root, dirs, files in os.walk(data_dir):
        for dir in dirs:
            print(f"Checking {dir}.")
            try:
                bin_dir = ifcb.DataDirectory(os.path.join(root, dir))
            except FileNotFoundError:
                print(f"Error: Data directory not found at '{bin_dir}'.")
            except Exception as e:
                print(f"Error loading data directory: {e}")

            for sample_bin in bin_dir:
                print(f"Saving {sample_bin.pid}.")
                try:
                    os.makedirs(f"{output_dir}/{sample_bin.pid}", exist_ok=True)
                except OSError as e:
                    print(f"Error creating output directory '{output_dir}': {e}")
                for number in sample_bin.images.keys():
                    try:
                        # This is where the ValueError usually happens
                        image = sample_bin.images[number] 
                        
                        # Process and save
                        img_array = image.astype(np.uint8) * 255
                        inverted_image = ImageOps.invert(Image.fromarray(img_array))
                        inverted_image.save(f"{output_dir}/{sample_bin.pid}/{number}.png")
                        total_images_saved += 1
                        
                    except ValueError as ve:
                        print(f"  [Skip] ROI {number} in {sample_bin.pid} has dimensions mismatch: {ve}")
                    except Exception as e:
                        print(f"  [Skip] ROI {number} in {sample_bin.pid} failed: {e}")
        
    print(f"Saved {total_images_saved} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract IFCB images to PNGs")
    parser.add_argument("data_directory",help="Path to the directory containing IFCB data.")
    parser.add_argument("output_directory", help="Path to the directory to save the output CSV file and blobs.")

    args = parser.parse_args()
    extract_images(args.data_directory,args.output_directory)

    