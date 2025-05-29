from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = None

input_folder = '../dol_images'  
output_folder = '../dol_compressed'  

input_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..','static' ,'dol_images')
output_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'static','dol_compressed')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)



def resize_image(img, max_dimension=3000):
    width, height = img.size
    max_side = max(width, height)

    if max_side > max_dimension:
        scale_factor = max_dimension / max_side
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img



def compress_image(image_path, output_path, max_size_mb=10):
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(image_path) as img:
            img = resize_image(img)
            img = img.convert('RGB')

            quality = 85
            while True:
                img.save(output_path, quality=quality)
                if os.path.getsize(output_path) <= max_size_mb * 1024 * 1024:
                    break
                quality -= 5

            print(
                f"Compressed: {image_path} -> {output_path} (size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB)")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

image_count = 0
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_count += 1
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        compress_image(image_path, output_path)

