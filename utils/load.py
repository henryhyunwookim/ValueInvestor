import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from numpy import asarray
from collections import defaultdict
from PIL import Image
from numpy import asarray
from tqdm import tqdm

def load_data(file_name, folder_name=None):
    parent_path = Path(os.getcwd()).parent
    if folder_name:
        # path = parent_path / folder_name / file_name
        path = Path("/".join([os.getcwd(), folder_name, file_name]))
    else:
        # path = parent_path / file_name
        path = Path("/".join([os.getcwd(), file_name]))
    
    file_type = file_name.split(".")[1]
    if file_type=="csv":
        output = pd.read_csv(path)
    elif file_type== "xlsx":
        output = pd.read_excel(path)
    else:
        raise f"Failed to load data - invalid file type {file_type}"
    
    print(output.info(), "\n")
    print(output.describe(), "\n")

    return output


def check_file_downloaded(file_name, default_path, download_path):
    os.chdir(download_path)
    if os.path.exists(file_name):
        print(f"File {file_name} exist in {download_path}!")
    else:
        print(f"File {file_name} doesn't exist in {download_path}!")

    os.chdir(default_path)


def extract_zip_file(file_path, download_path, file_name):
    if os.path.exists(download_path + "\\" + file_name.split(".")[0]):
        print(f"{file_name} already extracted in {download_path}.")
    else:
        with ZipFile(file_path, 'r') as zip_file:
            zip_file.extractall(path=download_path)
        print(f"{file_name} extracted in {download_path}.")


def load_images(download_path, as_array=False):
    files_dict = defaultdict(dict)
    for f1 in os.listdir(download_path):
        if f1 == "images":
            print(f"Loading files in {download_path}\\{f1}")
            for f2 in os.listdir(download_path + f"\\{f1}"): # training or testing
                if files_dict.get(f2, "") == "":
                    files_dict[f2] = defaultdict(dict)

                print(f"Loading files in {download_path}\\{f1}\\{f2}")
                for f3 in os.listdir(download_path + f"\\{f1}\\{f2}"): # flip or notflip
                    if files_dict.get(f3, "") == "":
                        files_dict[f3] = defaultdict(dict)
                    
                    print(f"Loading files in {download_path}\\{f1}\\{f2}\\{f3}")
                    for f4 in tqdm(os.listdir(download_path + f"\\{f1}\\{f2}\\{f3}")):
                        if files_dict.get(f4, "") == "":
                            files_dict[f4] = defaultdict(dict)
                        
                        # Load each image file and convert it into a 3d (RGB) array.
                        jpg_file_path = download_path + f"\\{f1}\\{f2}\\{f3}\\{f4}"
                        image = Image.open(jpg_file_path)
                        if as_array:
                            image_array = asarray(image)
                            files_dict[f2][f3][f4] = image_array
                        else:
                            files_dict[f2][f3][f4] = image

    return files_dict


def get_image_shape(array_dict):
    image_shape = None
    for k, v in array_dict.items():
        for k2, v2 in v.items():
            for k3, v3 in v2.items():
                while image_shape == None:
                    image_array = asarray(v3)
                    image_shape = image_array.shape
                    print(f"Image shape: {image_shape}")
    return image_shape