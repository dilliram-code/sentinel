from PIL import Image
import os

# Folder containing the images
folder = "data/stakeholders"

# Supported image extensions
supported_extensions = (".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".webp")

# Loop through all files in the folder
for filename in os.listdir(folder):
    if filename.lower().endswith(supported_extensions):
        input_path = os.path.join(folder, filename)

        # Create output filename with .jpg extension
        output_filename = os.path.splitext(filename)[0] + ".jpg"
        output_path = os.path.join(folder, output_filename)

        try:
            # Open image
            with Image.open(input_path) as img:
                # Convert to RGB (necessary for PNGs with transparency)
                rgb_img = img.convert("RGB")

                # Save as JPG
                rgb_img.save(output_path, "JPEG", quality=95)

            print(f" Converted: {filename} -> {output_filename}")

        except Exception as e:
            print(f" Failed to convert {filename}: {e}")

print("\n Conversion completed!")