import os
import re
import eyed3
from datetime import datetime

def extract_date_from_title(title):
    match = re.search(r'(\d{6})', title)
    if match:
        date_str = match.group(1)
        try:
            date_obj = datetime.strptime(date_str, '%d%m%y')
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            pass
    return None

def process_mp3_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.mp3'):
            file_path = os.path.join(folder_path, filename)

            # Load the MP3 file
            audiofile = eyed3.load(file_path)

            # Extract date from title
            if audiofile.tag and audiofile.tag.title:
                title = audiofile.tag.title
                new_date = extract_date_from_title(title)

                # If a valid date is found, update the filename
                if new_date:
                    new_filename = f"({new_date}) {filename}"
                    new_filepath = os.path.join(folder_path, new_filename)

                    # Rename the file
                    os.rename(file_path, new_filepath)
                    print(f"Renamed: {filename} -> {new_filename}")

def remove_date_prefix(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.mp3'):
            file_path = os.path.join(folder_path, filename)

            # Check if the filename starts with the specified date format
            match = re.match(r'\(\d{2}-\d{2}-\d{4}\) ', filename)
            if match:
                # Remove the date prefix
                new_filename = filename[match.end():]
                new_filepath = os.path.join(folder_path, new_filename)

                # Rename the file
                os.rename(file_path, new_filepath)
                print(f"Removed date prefix: {filename} -> {new_filename}")


if __name__ == "__main__":
    folder_path = "C:\\Users\\Luis\\Documents\\Wizards of the Coast\\Magic the Gathering\\profiles\\Audio\\lorenzo ramirez\\2011"
    process_mp3_files(folder_path)
    #remove_date_prefix(folder_path)