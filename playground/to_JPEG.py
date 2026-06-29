from PIL import Image
import os

# Folder containing the images
folder = "data/stakeholders"

# Image formats that can be converted
convertible_extensions = (".png", ".bmp", ".tiff", ".tif", ".webp", ".gif")

for filename in os.listdir(folder):
    input_path = os.path.join(folder, filename)

    # Skip directories
    if not os.path.isfile(input_path):
        continue

    name, ext = os.path.splitext(filename)
    ext = ext.lower()

    # Keep existing JPG/JPEG images untouched
    if ext in (".jpg", ".jpeg"):
        print(f"Keeping existing JPG: {filename}")
        continue

    # Process other supported image formats
    if ext in convertible_extensions:
        output_filename = f"{name}.jpg"
        output_path = os.path.join(folder, output_filename)

        try:
            # If JPG already exists, remove the duplicate image
            if os.path.exists(output_path):
                os.remove(input_path)
                print(f"Removed duplicate: {filename} (kept {output_filename})")
                continue

            # Convert image to JPG
            with Image.open(input_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(output_path, "JPEG", quality=95)

            # Delete the original image
            os.remove(input_path)

            print(f"Converted: {filename} -> {output_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("\nFinished! Only JPG images remain.")