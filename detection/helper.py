import os
import requests
import zipfile


def download_file_if_not_exists(url, directory, filename):
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Full path to the file
    file_path = os.path.join(directory, filename)
    
    # Check if file already exists
    if not os.path.isfile(file_path):
        print(f"File not found. Downloading {filename} from {url}...")
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Write the content to the file
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"{filename} has been downloaded and saved in {directory}.")
            # Unzip the file if it is a zip file
            if zipfile.is_zipfile(file_path):
                print(f"Unzipping {filename}...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(directory)
                print(f"{filename} has been unzipped in {directory}.")
            else:
                print(f"{filename} is not a valid zip file.")
                else:
            print(f"Failed to download the file. Status code: {response.status_code}")
    else:
        print(f"{filename} already exists in {directory}.")

    


