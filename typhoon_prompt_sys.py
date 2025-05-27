PROMPTS_SYS = {
    "default": lambda base_text: (
        f"Below is an image of a land deed document, along with its dimensions and possibly some raw textual content previously extracted from it. "
        f"Your task is to extract and return the following features from the document:\n"
        f"1. ที่ดินระวาง: Alphanumeric code, Format: ####/##, E.g., '1234/56'\n"
        f"2. เลขที่ดิน: Numeric, Range: 0-999999\n"
        f"3. ตำบล: Text, E.g., 'บางขุนเทียน'\n"
        f"4. อำเภอ: Text, E.g., 'เมืองนนทบุรี'\n"
        f"5. ฉโนดที่ดิน: Numeric, Range: 0-999\n"
        f"6. เลขเล่ม: Numeric, Range: 0-99\n"
        f"7. หน้าสำรวจ: Numeric, Range: 0-100\n"
        f"8. มาตราส่วน / สเกล: Text or Numeric ratio, E.g., '1:500', format: 1:###\n"
        f"9. ลงชื่อ: Text or Image, E.g., 'นายสมชาย'\n"
        f"10. ผู้เขียนแผนที่: Text, E.g., 'บริษัทแผนที่บางขุนเทียน'\n\n"
        f"Please extract the above features from the following OCR-extracted text from the land deed:\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END\n\n"
        f"Your final output must be in JSON format with each feature key and its extracted value. The response should contain a single key 'extracted_data' with the extracted values in the following format:\n"
        f"{{\n"
        f"  \"ที่ดินระวาง\": \"<value>\",\n"
        f"  \"เลขที่ดิน\": \"<value>\",\n"
        f"  \"ตำบล\": \"<value>\",\n"
        f"  \"อำเภอ\": \"<value>\",\n"
        f"  \"ฉโนดที่ดิน\": \"<value>\",\n"
        f"  \"เลขเล่ม\": \"<value>\",\n"
        f"  \"หน้าสำรวจ\": \"<value>\",\n"
        f"  \"มาตราส่วน / สเกล\": \"<value>\",\n"
        f"  \"ลงชื่อ\": \"<value>\",\n"
        f"  \"ผู้เขียนแผนที่\": \"<value>\"\n"
        f"}}"
    ),
    
    "structure": lambda base_text: (
        f"Below is an image of a land deed document, along with its dimensions and possibly some raw textual content previously extracted from it. "
        f"Your task is to carefully extract and return the following features from the document:\n"
        f"1. ที่ดินระวาง: Alphanumeric code, Format: ####/##\n"
        f"2. เลขที่ดิน: Numeric\n"
        f"3. ตำบล: Text\n"
        f"4. อำเภอ: Text\n"
        f"5. ฉโนดที่ดิน: Numeric\n"
        f"6. เลขเล่ม: Numeric\n"
        f"7. หน้าสำรวจ: Numeric\n"
        f"8. มาตราส่วน / สเกล: Numeric ratio\n"
        f"9. ลงชื่อ: Text or Image\n"
        f"10. ผู้เขียนแผนที่: Text\n\n"
        f"Please extract the above features from the following OCR-extracted text from the land deed:\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END\n\n"
        f"Your response should strictly contain the JSON output with the values for each feature. Ensure the output is in the following JSON format with no additional text:\n"
        f"{{\n"
        f"  \"ที่ดินระวาง\": \"<value>\",\n"
        f"  \"เลขที่ดิน\": \"<value>\",\n"
        f"  \"ตำบล\": \"<value>\",\n"
        f"  \"อำเภอ\": \"<value>\",\n"
        f"  \"ฉโนดที่ดิน\": \"<value>\",\n"
        f"  \"เลขเล่ม\": \"<value>\",\n"
        f"  \"หน้าสำรวจ\": \"<value>\",\n"
        f"  \"มาตราส่วน / สเกล\": \"<value>\",\n"
        f"  \"ลงชื่อ\": \"<value>\",\n"
        f"  \"ผู้เขียนแผนที่\": \"<value>\"\n"
        f"}}"
    ),
}

import os
import traceback
from PIL import Image
from typhoon_ocr import ocr_document

# Get absolute path to current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output folders
image_folder = os.path.join(script_dir, 'images')
output_folder = os.path.join(script_dir, 'research_results', 'typhoon_ocr_results')

# Set Typhoon OCR API key
os.environ['TYPHOON_OCR_API_KEY'] = "sk-FsXJJmQBEw6YLmT1cM2TL5Tbo4bTRYmrZTaDZxqXNXMCrrtV"

# Disable image pixel limit warning
Image.MAX_IMAGE_PIXELS = None

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)


def process_and_store_result(image_file_path, output_file_path):
    try:
        if not os.path.exists(image_file_path):
            print(f"❌ File not found: {image_file_path}")
            return

        try:
            with Image.open(image_file_path) as img:
                print(f"✅ Opened image: {image_file_path} | Format: {img.format}")
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
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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
