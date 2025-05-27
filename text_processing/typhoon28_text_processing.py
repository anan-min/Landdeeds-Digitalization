import os
import traceback
import json
import ollama


model_name = "typhoon28"
# Define input and output folders
output_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'research_results', 'typhoon_ocr_results')
processed_folder = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'research_results', f'{model_name}_text_processed')

features = {
    "ที่ดินระวาง": "Alphanumeric code, Format: ####/##, E.g., '1234/56'",
    "เลขที่ดิน": "Numeric, Range: 0-999999",
    "ตำบล": "Text, E.g., 'บางขุนเทียน'",
    "อำเภอ": "Text, E.g., 'เมืองนนทบุรี'",
    "ฉโนดที่ดิน": "Numeric, Range: 0-999",
    "เลขเล่ม": "Numeric, Range: 0-99",
    "หน้าสำรวจ": "Numeric, Range: 0-100",
    "มาตราส่วน / สเกล": "Text or Numeric ratio, E.g., '1:500', format: 1:###",
    "ลงชื่อ": "Text or Image, E.g., 'นายสมชาย'",
    "ผู้เขียนแผนที่": "Text, E.g., 'บริษัทแผนที่บางขุนเทียน'"
}


# Ensure processed folder exists
os.makedirs(processed_folder, exist_ok=True)


def process_and_store_result(txt_file_path, output_file_path):
    try:
        if not os.path.exists(txt_file_path):
            print(f"❌ File not found: {txt_file_path}")
            return

        try:
            with open(txt_file_path, 'r', encoding='utf-8') as file:
                print(f"✅ Opened text file: {txt_file_path}")
                text_content = file.read()
                print("🧠 Processing text content...")
                # Add your text processing logic here. For now, it will just clean up text:
                stripped_text = text_content.strip()
                prompt = generatePrompt(stripped_text)
                processed_text = run_ollama_model(
                    model_name="scb10x/llama3.1-typhoon2-8b-instruct",
                    prompt=prompt
                )

        except Exception as e:
            print(f"❌ Failed to read text file: {txt_file_path} | Error: {e}")
            return

        # Save the processed text to the output folder
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(processed_text)

        print(f"💾 Saved processed result to: {output_file_path}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()


def process_all_texts_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Text folder does not exist: {folder_path}")
        return

    try:
        txt_files = [f for f in os.listdir(
            folder_path) if f.lower().endswith('.txt')]
        print(f"📄 Found {len(txt_files)} text files to process.")

        for txt_file in txt_files:
            txt_path = os.path.join(folder_path, txt_file)
            base_name = os.path.splitext(txt_file)[0]
            output_file_path = os.path.join(
                processed_folder, f"{base_name}_processed.txt")

            process_and_store_result(txt_path, output_file_path)

    except Exception as e:
        print(f"❌ Error scanning folder: {e}")
        traceback.print_exc()


def generatePrompt(input_text):
    prompt = "Given the following list of features, please extract the corresponding values from the input text. Return the values as JSON.\n\n"
    prompt += "Features: " + \
        json.dumps(features, ensure_ascii=False, indent=4) + "\n\n"
    prompt += "Input text: " + input_text + "\n\n"
    prompt += "Only provide the output in JSON format, with no additional explanation.\n"
    prompt += "Do not include any text before or after the JSON. Only return the JSON."
    return prompt


def run_ollama_model(model_name, prompt):
    result = ollama.generate(model=model_name, prompt=prompt)
    return result['response']


if __name__ == "__main__":
    print("📁 Output folder:", output_folder)
    print("📁 Processed folder:", processed_folder)
    process_all_texts_in_folder(output_folder)
