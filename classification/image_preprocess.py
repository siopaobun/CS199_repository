import os
from PIL import Image, ImageOps

input_folder = "CS199_repository/image_extraction/output_pngs/D20250617T011402_IFCB202/" 
output_folder = "inverted_images" 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
file_count = 0
skipped_count = 0

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename) 

        try:
            with Image.open(input_path) as img:
                final_img = ImageOps.invert(img.convert('RGB'))

                final_img.save(output_path)
                file_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped_count += 1
    else:
        pass

print(f"Successfully processed {file_count} images.")
if skipped_count > 0:
    print(f"Skipped {skipped_count} files due to errors.")