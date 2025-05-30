import os
import traceback
from PIL import Image
from typhoon_ocr import ocr_document

script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(script_dir, '..', 'static', 'dol_compressed')
# static/dol_compressed
output_folder = os.path.join(
    script_dir, '..', 'research_results', 'ocr_results', 'typhoonOCR_newDOL_results')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

os.environ['TYPHOON_OCR_API_KEY'] = "sk-FsXJJmQBEw6YLmT1cM2TL5Tbo4bTRYmrZTaDZxqXNXMCrrtV"
Image.MAX_IMAGE_PIXELS = None
os.makedirs(output_folder, exist_ok=True)


def process_and_store_result(image_file_path, output_file_path):
    try:
        if not os.path.exists(image_file_path):
            print(f"❌ File not found: {image_file_path}")
            return

        try:
            with Image.open(image_file_path) as img:
                print(
                    f"✅ Opened image: {image_file_path} | Format: {img.format}")
        except Exception as e:
            print(f"❌ Failed to open image: {image_file_path} | Error: {e}")
            return

        print(f"🧠 Running OCR on: {image_file_path}")
        try:
            markdown = ocr_document(image_file_path, task_type="default")
        except Exception as e:
            print(f"❌ OCR failed on {image_file_path}: {e}")
            traceback.print_exc()
            return

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"💾 Saved result to: {output_file_path}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()


def process_all_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Image folder does not exist: {folder_path}")
        return

    try:
        image_files = [f for f in os.listdir(
            folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"📄 Found {len(image_files)} images to process.")

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            base_name = os.path.splitext(image_file)[0]
            output_file_path = os.path.join(output_folder, f"{base_name}.txt")

            process_and_store_result(image_path, output_file_path)

    except Exception as e:
        print(f"❌ Error scanning folder: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("📁 Script directory:", script_dir)
    print("📁 Image folder:", image_folder)
    print("📁 Output folder:", output_folder)
    process_all_images_in_folder(image_folder)
