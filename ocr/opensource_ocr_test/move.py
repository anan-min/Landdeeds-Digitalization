# import shutil
# import os

# # Path to the source folder where `tha.traineddata` is located
# # Current directory (or adjust if the file is in a different folder)
# source_folder = './'
# # Path to the destination folder
# destination_folder = '/opt/homebrew/share/tessdata/'

# # Name of the language file to move
# language_file = 'tha.traineddata'

# # Full source and destination paths
# source_path = os.path.join(source_folder, language_file)
# destination_path = os.path.join(destination_folder, language_file)

# try:
#     # Check if the file exists in the source folder
#     if os.path.exists(source_path):
#         # Move the file to the destination folder
#         shutil.move(source_path, destination_path)
#         print(f"Successfully moved {language_file} to {destination_folder}")
#     else:
#         print(f"{language_file} does not exist in the source folder.")
# except Exception as e:
#     print(f"Error moving {language_file}: {e}")


import os

# Path to the tessdata directory
tessdata_path = '/opt/homebrew/share/tessdata/'

# File name of the Thai language trained data
language_file = 'tha.traineddata'

# Full path to the language file
language_file_path = os.path.join(tessdata_path, language_file)

# Check if the file exists
if os.path.exists(language_file_path):
    print(f"{language_file} is found at {language_file_path}")
else:
    print(f"{language_file} is NOT found at {language_file_path}")
