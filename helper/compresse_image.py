from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = None

# Set the folder paths
input_folder = '../dol'  # Folder where original images are stored
output_folder = '../images'  # Folder where compressed images will be stored

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to resize the image while maintaining aspect ratio


def resize_image(img, max_dimension=3000):
    width, height = img.size
    max_side = max(width, height)

    if max_side > max_dimension:
        # Calculate the scale ratio to resize the image
        scale_factor = max_dimension / max_side
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img

# Function to compress images


def compress_image(image_path, output_path, max_size_mb=10):
    try:
        # Open an image file
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(image_path) as img:
            # Resize the image if necessary
            img = resize_image(img)
            # Convert image to RGB (removes alpha if present)
            img = img.convert('RGB')

            # Save the image with quality control
            quality = 85
            while True:
                img.save(output_path, quality=quality)
                # Check file size
                if os.path.getsize(output_path) <= max_size_mb * 1024 * 1024:
                    break
                # Reduce quality if file size is too large
                quality -= 5

            print(
                f"Compressed: {image_path} -> {output_path} (size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB)")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Full path to the image
        image_path = os.path.join(input_folder, filename)
        # Set the output path for compressed image
        output_path = os.path.join(output_folder, filename)
        # Compress the image
        compress_image(image_path, output_path)
